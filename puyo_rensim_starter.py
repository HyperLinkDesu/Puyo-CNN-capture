import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import mss
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False


CONFIG_PATH = "puyo_config.json"
DEFAULT_GRID = (6, 12)  # columns, rows — adjust for your game/skin
CELL_PAD = 2  # pixels to shrink cell crop to avoid borders

LABELS = ["empty", "red", "blue", "green", "yellow", "purple", "garbage"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}

HSV_THRESHOLDS = {
    "red":    [(0, 70, 50), (10, 255, 255)],
    "red2":   [(170, 70, 50), (180, 255, 255)],
    "blue":   [(90, 50, 50), (130, 255, 255)],
    "green":  [(45, 50, 50), (85, 255, 255)],
    "yellow": [(15, 50, 50), (35, 255, 255)],
    "purple": [(130, 40, 40), (170, 255, 255)],
}


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


def select_roi_from_image(img: np.ndarray) -> ROI:
    clone = img.copy()
    r = cv2.selectROI("Select board ROI", clone, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board ROI")
    x, y, w, h = map(int, r)
    return ROI(x, y, w, h)


def auto_detect_roi(img: np.ndarray, expected_grid=DEFAULT_GRID) -> ROI:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, th = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    conts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        h, w = img.shape[:2]
        return ROI(0, 0, w, h)
    c = max(conts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    padx = int(w * 0.05)
    pady = int(h * 0.05)
    x = max(0, x - padx)
    y = max(0, y - pady)
    w = min(img.shape[1] - x, w + 2 * padx)
    h = min(img.shape[0] - y, h + 2 * pady)
    return ROI(x, y, w, h)


def crop_grid_cells(img: np.ndarray, roi: ROI, grid: Tuple[int,int]=DEFAULT_GRID) -> List[np.ndarray]:
    cols, rows = grid
    cell_w = roi.w / cols
    cell_h = roi.h / rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(roi.x + c * cell_w + CELL_PAD)
            y1 = int(roi.y + r * cell_h + CELL_PAD)
            x2 = int(roi.x + (c+1) * cell_w - CELL_PAD)
            y2 = int(roi.y + (r+1) * cell_h - CELL_PAD)
            cell = img[y1:y2, x1:x2]
            if cell.size == 0:
                cell = np.zeros((int(cell_h)-2*CELL_PAD, int(cell_w)-2*CELL_PAD, 3), np.uint8)
            cells.append(cell)
    return cells


def classify_cell_hsv(cell: np.ndarray) -> str:
    """
    Fast rule-based HSV classifier. Returns label string.
    """
    if cell is None or cell.size == 0:
        return "empty"
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cx1, cy1 = int(w*0.2), int(h*0.2)
    cx2, cy2 = int(w*0.8), int(h*0.8)
    patch = hsv[cy1:cy2, cx1:cx2]
    mean_s = float(np.mean(patch[:,:,1]))
    mean_v = float(np.mean(patch[:,:,2]))
    if mean_s < 30 or mean_v < 30:
        return "empty"
    for label, rng in HSV_THRESHOLDS.items():
        if label == 'red2':
            lower, upper = np.array(rng[0]), np.array(rng[1])
            mask = cv2.inRange(patch, lower, upper)
            if np.count_nonzero(mask) > mask.size * 0.05:
                return "red"
            continue
        lower, upper = np.array(rng[0]), np.array(rng[1])
        mask = cv2.inRange(patch, lower, upper)
        if np.count_nonzero(mask) > mask.size * 0.06:
            if label == 'red' or label == 'red2':
                return 'red'
            return label
    avg_h = np.mean(patch[:,:,0])
    if avg_h < 15 or avg_h > 165:
        return 'red'
    if 15 <= avg_h < 40:
        return 'yellow'
    if 40 <= avg_h < 90:
        return 'green'
    if 90 <= avg_h < 130:
        return 'blue'
    if 130 <= avg_h < 165:
        return 'purple'
    return 'empty'


def collect_dataset_from_image(img_path: str, roi: ROI, grid=DEFAULT_GRID, out_dir='dataset') -> None:
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    cells = crop_grid_cells(img, roi, grid)
    print(f"Saving {len(cells)} cells. Press label key for each cell. Keys: e r b g y p x (garbage) or s to skip")
    i = 0
    for cell in cells:
        cv2.imshow('cell', cv2.resize(cell, (128,128), interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            i += 1
            continue
        lbl = None
        if k == ord('e'):
            lbl = 'empty'
        elif k == ord('r'):
            lbl = 'red'
        elif k == ord('b'):
            lbl = 'blue'
        elif k == ord('g'):
            lbl = 'green'
        elif k == ord('y'):
            lbl = 'yellow'
        elif k == ord('p'):
            lbl = 'purple'
        elif k == ord('x'):
            lbl = 'garbage'
        else:
            print('unknown key, skipping')
            i += 1
            continue
        fname = os.path.join(out_dir, f"{i:05d}_{lbl}.png")
        cv2.imwrite(fname, cell)
        print('saved', fname)
        i += 1
    cv2.destroyAllWindows()


if TORCH_AVAILABLE:
    class PuyoCellDataset(Dataset):
        def __init__(self, folder, transform=None):
            self.samples = []
            self.transform = transform
            for fn in os.listdir(folder):
                if not fn.lower().endswith('.png'):
                    continue
                label = fn.split('_')[-1].split('.')[0]
                if label not in LABEL_TO_INT:
                    continue
                self.samples.append((os.path.join(folder, fn), LABEL_TO_INT[label]))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            fp, label = self.samples[idx]
            img = Image.open(fp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    class TinyPuyoNet(nn.Module):
        def __init__(self, n_classes=len(LABELS)):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, n_classes)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def train_cnn(dataset_dir='dataset', epochs=10, batch_size=64, lr=1e-3, out='puyo_cnn.pth'):
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        ds = PuyoCellDataset(dataset_dir, transform=transform)
        if len(ds) == 0:
            print('No data in', dataset_dir)
            return
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = TinyPuyoNet()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for ep in range(epochs):
            tot_loss = 0.0
            correct = 0
            seen = 0
            for xb, yb in dl:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += float(loss) * xb.size(0)
                correct += int((preds.argmax(dim=1) == yb).sum())
                seen += xb.size(0)
            print(f"Epoch {ep+1}/{epochs} loss={tot_loss/seen:.4f} acc={correct/seen:.3f}")
        torch.save(model.state_dict(), out)
        print('Saved', out)

    def load_cnn_model(path='puyo_cnn.pth'):
        model = TinyPuyoNet()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

    def classify_cell_cnn(cell: np.ndarray, model) -> str:
        img = cv2.resize(cell, (32,32), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            lbl = int(out.argmax(dim=1).item())
        return LABELS[lbl]


def board_to_rensim_string(labels: List[str], grid=DEFAULT_GRID) -> str:
    cols, rows = grid
    lines = []
    for r in range(rows):
        row_labels = labels[r*cols:(r+1)*cols]
        lines.append(','.join(row_labels))
    return '\n'.join(lines)


def copy_to_clipboard(text: str):
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True


def autofill_rensim(labels: List[str], grid=DEFAULT_GRID):
    if not PYAUTOGUI_AVAILABLE:
        print('pyautogui not available; cannot autofill')
        return
    print('Autofill will start in 3 seconds. Make sure rensim import UI is focused and top-left cell is active.')
    time.sleep(3)
    cols, rows = grid
    for r in range(rows):
        for c in range(cols):
            lbl = labels[r*cols + c]
            if lbl == 'empty':
                pyautogui.press('backspace')
            else:
                mapping = {'red':'r','blue':'b','green':'g','yellow':'y','purple':'p','garbage':'x'}
                key = mapping.get(lbl, '')
                if key:
                    pyautogui.typewrite(key)
            pyautogui.press('tab')
        pyautogui.press('enter')


class App:
    def __init__(self, master):
        self.master = master
        master.title('Puyo → Rensim Helper')
        self.cfg = load_config()
        self.grid = tuple(self.cfg.get('grid', DEFAULT_GRID))
        self.roi = None
        self.model = None

        self.img_label = tk.Label(master, text='No image')
        self.img_label.pack()

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text='Load image', command=self.load_image).pack(side='left')
        tk.Button(btn_frame, text='Select ROI', command=self.select_roi).pack(side='left')
        tk.Button(btn_frame, text='Auto-detect ROI', command=self.auto_roi).pack(side='left')
        tk.Button(btn_frame, text='Detect board', command=self.detect_board).pack(side='left')
        tk.Button(btn_frame, text='Export rensim', command=self.export_rensim).pack(side='left')
        tk.Button(btn_frame, text='Collect dataset', command=self.collect_dataset).pack(side='left')
        tk.Button(btn_frame, text='Train CNN', command=self.train_cnn).pack(side='left')

        self.image = None
        self.last_labels = None

    def load_image(self):
        fp = filedialog.askopenfilename(filetypes=[('Images','*.png;*.jpg;*.bmp')])
        if not fp:
            return
        img = cv2.imread(fp)
        self.image = img
        self.show_preview(img)

    def show_preview(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = min(640/w, 480/h, 1.0)
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tkimg = ImageTk.PhotoImage(pil)
        self.img_label.configure(image=tkimg)
        self.img_label.image = tkimg

    def select_roi(self):
        if self.image is None:
            messagebox.showinfo('Info', 'Load an image first')
            return
        self.roi = select_roi_from_image(self.image)
        self.cfg['roi'] = [self.roi.x, self.roi.y, self.roi.w, self.roi.h]
        save_config(self.cfg)
        messagebox.showinfo('Info', f'ROI saved: {self.cfg["roi"]}')

    def auto_roi(self):
        if self.image is None:
            messagebox.showinfo('Info', 'Load an image first')
            return
        self.roi = auto_detect_roi(self.image, self.grid)
        self.cfg['roi'] = [self.roi.x, self.roi.y, self.roi.w, self.roi.h]
        save_config(self.cfg)
        messagebox.showinfo('Info', f'Auto ROI saved: {self.cfg["roi"]}')

    def detect_board(self):
        if self.image is None:
            messagebox.showinfo('Info', 'Load an image first')
            return
        if self.roi is None:
            if 'roi' in self.cfg:
                x, y, w, h = self.cfg['roi']
                self.roi = ROI(x, y, w, h)
            else:
                messagebox.showinfo('Info', 'Select ROI first')
                return

        cells = crop_grid_cells(self.image, self.roi, self.grid)

        cols, rows = self.grid
        labels = []
        idx = 0

        use_cnn = TORCH_AVAILABLE and os.path.exists('puyo_cnn.pth')
        if use_cnn and self.model is None:
            self.model = load_cnn_model('puyo_cnn.pth')

        for r in range(rows):
            for c in range(cols):
                cell = cells[idx]

                if c == 2:
                    labels.append("empty")
                    idx += 1
                    continue

                if use_cnn:
                    labels.append(classify_cell_cnn(cell, self.model))
                else:
                    labels.append(classify_cell_hsv(cell))

                idx += 1

        self.last_labels = labels
        messagebox.showinfo('Detected', 'Board detected — now export or review')
        self.show_cells_preview(cells, labels)


    def show_cells_preview(self, cells, labels):
        cols, rows = self.grid
        cell_h = 40
        cell_w = 40
        mosaic = np.zeros((rows*cell_h, cols*cell_w, 3), dtype=np.uint8)+40
        idx = 0
        for r in range(rows):
            for c in range(cols):
                cell = cells[idx]
                if cell.size:
                    small = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                else:
                    small = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(small, labels[idx][0], (2,cell_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                mosaic[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = small
                idx += 1
        self.show_preview(mosaic)

    def export_rensim(self):
        if not self.last_labels:
            messagebox.showinfo('Info', 'Detect board first')
            return
        s = board_to_rensim_string(self.last_labels, self.grid)
        copy_to_clipboard(s)
        fp = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text','*.txt')])
        if fp:
            with open(fp, 'w') as f:
                f.write(s)
        messagebox.showinfo('Exported', 'Exported to clipboard and file (if chosen)')

    def collect_dataset(self):
        if self.image is None:
            messagebox.showinfo('Info', 'Load an image first')
            return
        if 'roi' not in self.cfg:
            messagebox.showinfo('Info', 'Select ROI first')
            return

        roi = ROI(*self.cfg['roi'])
        cells = crop_grid_cells(self.image, roi, self.grid)

        os.makedirs('dataset', exist_ok=True)
        print(f"Saving {len(cells)} cells. Press label key: e r b g y p x, or s to skip")
        
        cols, rows = self.grid
        i = 0
        idx = 0

        for r in range(rows):
            for c in range(cols):
                cell = cells[idx]

                if c == 2:  # 3rd slot is always empty
                    lbl = "empty"
                    fname = os.path.join("dataset", f"{idx:05d}_{lbl}.png")
                    cv2.imwrite(fname, cell)
                    print("Auto-saved 3rd column as empty:", fname)
                    idx += 1
                    continue

                cv2.imshow('cell', cv2.resize(cell, (128, 128), interpolation=cv2.INTER_NEAREST))
                k = cv2.waitKey(0) & 0xFF

                label_map = {
                    ord('e'): 'empty',
                    ord('r'): 'red',
                    ord('b'): 'blue',
                    ord('g'): 'green',
                    ord('y'): 'yellow',
                    ord('p'): 'purple',
                    ord('x'): 'garbage'
                }

                if k == ord('s'):
                    cv2.destroyWindow('cell')
                    idx += 1
                    continue

                if k not in label_map:
                    print("Unknown key, skipping")
                    cv2.destroyWindow('cell')
                    idx += 1
                    continue

                lbl = label_map[k]
                fname = os.path.join("dataset", f"{idx:05d}_{lbl}.png")
                cv2.imwrite(fname, cell)
                print("Saved", fname)

                cv2.destroyWindow('cell')
                idx += 1




    def train_cnn(self):
        if not TORCH_AVAILABLE:
            messagebox.showinfo('Info', 'PyTorch not available')
            return
        train_cnn()


def capture_screen_save(fp='screenshot.png'):
    if not MSS_AVAILABLE:
        print('mss not available; cannot capture screen')
        return
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img.save(fp)
        print('Saved', fp)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
