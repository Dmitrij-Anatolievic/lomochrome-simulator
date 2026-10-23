#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  LOMOCHROME SIMULATOR v1                                         ║
║──────────────────────────────────────────────────────────────────║
║  Pellicole: Purple · Turquoise · Metropolis                      ║
║             Color '92 · Color '92 Sun-kissed · Classicolor        ║
║                                                                  ║
║  Purple / Turquoise / Metropolis — modelli MISURATI:             ║
║   tavola ColorChecker 24 patch su pellicola (Purple, Turquoise), ║
║   stessa scena in digitale e su pellicola (Lomography, 2022),    ║
║   fotogrammi allineati al pixel, coppie colore digitale→film,    ║
║   fit di una matrice di scambio dei canali (come gli strati      ║
║   sensibili invertiti dell'emulsione) + curve tonali per canale, ║
║   con esposizione separata per ogni scena, + correzione locale   ║
║   RBF sulle patch; Metropolis vincolata su oggetti noti.         ║
║  Color '92 / Sun-kissed / Classicolor — impronta STATISTICA:     ║
║   dominante per fascia tonale, saturazione e contrasto misurati  ║
║   su 15–32 scansioni pubblicate da Lomography per pellicola.     ║
║                                                                  ║
║  Autore: Dmitrij Musella                                         ║
║  Engine: NumPy + Pillow + PyQt6                                  ║
╚══════════════════════════════════════════════════════════════════╝

Requisiti:  pip install PyQt6 Pillow numpy
Uso:        python lomochrome_sim.py
"""

import sys, os, math, zlib, random
from dataclasses import dataclass, replace, field
from functools import lru_cache
from pathlib import Path

if sys.platform == "win32":
    _wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(_wf):
        os.environ["QT_QPA_FONTDIR"] = _wf

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QFileDialog, QGroupBox,
    QComboBox, QSplitter, QListWidget, QSizePolicy, QProgressBar,
    QMessageBox, QAbstractItemView, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QImage, QPixmap, QAction, QColor, QPainter, QPen, QPalette

Image.MAX_IMAGE_PIXELS = 300_000_000

# ═══════════════════════════════════════════════════════════════
# PROFILI PELLICOLA
# ═══════════════════════════════════════════════════════════════

K_CURVE = 7   # nodi delle curve tonali per canale (modelli misurati)

# Parametri fittati: 9 (matrice 3×3) + 3 (offset) + 3×7 (curve monotone, log-incrementi)
MEASURED = {
    "purple": (0.71510, 0.26119, 0.24139, 0.05843, 0.08156, 0.51829, 0.04552, 0.43894, 0.35858, 0.11667, 0.11263, 0.01664, -0.30182, -1.86386, -1.94132, -1.64073, -2.11385, -0.93522, -1.81575, -0.24327, -1.85394, -2.00289, -1.55827, -1.06510, -1.89856, -1.75772, -0.08306, -1.82106, -2.03526, -1.55149, -1.56473, -2.09887, -2.05043),
    "turquoise": (-0.00010, -0.23930, 1.18000, -0.04864, 0.86715, 0.34803, 0.61469, 0.33471, 0.73327, 0.00106, -0.00768, 0.00175, -0.02589, -1.82864, -1.86414, -1.52345, -1.37060, -1.78627, -1.83014, 0.09497, -1.76425, -1.74194, -1.70594, -1.91487, -2.02460, -1.92689, -0.00171, -1.79069, -1.72973, -1.83088, -1.87865, -1.94359, -1.93226),
    "metropolis": (0.74810, 0.54609, 0.23575, 0.23493, 0.89970, 0.37636, 0.15687, 0.32408, 0.76493, 0.03978, 0.03320, 0.03652, -0.19374, -1.83216, -1.86709, -1.82937, -1.94539, -1.35488, -1.97480, -0.16339, -1.87114, -1.89099, -1.75524, -1.72543, -1.66997, -2.05360, -0.18713, -1.82850, -1.87364, -1.82344, -1.50315, -1.58110, -2.29803),
}

# Correzione residua locale (RBF gaussiano nello spazio sRGB esposto), tarata sulle
# 24 patch ColorChecker fotografate da Lomography + coppie di Vienna.
RESIDUAL = {
    "turquoise": dict(sigma=58.0, centers=[[61.0,120.0,204.0],[67.0,124.0,209.0],[79.0,134.0,218.0],[91.0,146.0,232.0],[103.0,153.0,232.0],[119.0,178.0,239.0],[136.0,183.0,235.0],[139.0,176.0,210.0],[181.0,199.0,208.0],[208.0,212.0,211.0],[120.0,168.0,216.0],[232.0,233.0,232.0],[158.0,162.0,168.0],[156.0,158.0,162.0],[104.0,151.0,208.0],[170.0,171.0,173.0],[111.0,111.0,113.0],[91.0,100.0,109.0],[107.0,140.0,179.0],[83.0,91.0,101.0],[146.0,205.0,241.0],[125.0,129.0,137.0],[78.0,79.0,79.0],[120.0,124.0,132.0],[176.0,215.0,229.0],[223.0,226.0,222.0],[208.0,222.0,226.0],[143.0,144.0,146.0],[172.0,164.0,155.0],[131.0,124.0,114.0],[58.0,62.0,67.0],[61.0,71.0,82.0],[217.0,226.0,227.0],[51.0,52.0,44.0],[76.0,76.0,53.0],[139.0,134.0,114.0],[99.0,94.0,89.0],[111.0,109.0,82.0],[66.0,63.0,55.0],[23.0,26.0,19.0],[167.0,157.0,144.0],[40.0,43.0,28.0],[62.0,65.0,43.0],[30.0,35.0,23.0],[132.0,126.0,92.0],[138.0,130.0,93.0],[42.3,85.5,173.7],[65.8,109.9,197.1],[52.6,97.7,188.7],[43.2,87.3,180.3],[52.6,93.0,172.7],[70.5,103.3,168.1],[102.4,108.0,123.0],[54.5,82.7,135.2],[72.3,93.9,135.2],[134.3,135.2,137.1],[138.0,145.5,155.9],[163.4,163.4,164.3],[93.0,116.4,163.4],[106.1,127.7,169.0],[96.7,97.7,106.1],[70.5,71.4,78.0],[77.0,82.7,95.8],[52.6,69.5,102.4],[56.4,62.0,74.2],[122.1,116.4,109.9],[51.7,55.4,64.8],[195.3,192.5,191.5],[86.4,92.0,105.2],[174.6,176.5,181.2],[211.2,205.6,201.8],[143.7,152.1,159.6],[181.2,177.4,173.7],[186.8,182.1,177.4],[151.2,146.5,141.8],[82.7,123.0,196.2],[154.9,153.0,146.5],[67.6,86.4,123.9],[63.9,61.1,58.3],[46.1,47.0,50.8],[83.6,135.2,231.9],[146.5,151.2,146.5],[60.1,58.3,57.3],[27.3,32.0,41.4],[23.5,27.3,33.9],[126.8,122.1,115.5],[66.7,115.5,210.3],[60.1,59.2,61.1],[18.8,21.7,27.3],[90.2,88.3,93.9],[92.0,141.8,235.6],[234.3,242.4,246.5],[222.1,234.3,243.4],[209.8,224.1,237.3],[211.9,217.0,219.0],[188.5,200.7,212.9],[172.2,180.3,188.5],[127.4,138.6,151.8],[179.3,188.5,200.7],[112.2,84.7,74.5],[80.6,54.2,47.0],[141.7,153.9,168.1],[80.6,77.6,78.6],[91.8,68.4,59.3],[61.3,68.4,75.5],[51.1,47.0,46.0],[57.2,62.3,68.4],[95.9,102.0,109.1],[108.1,110.1,115.2],[104.0,71.5,59.3],[115.2,125.4,134.6],[135.6,143.7,154.9],[91.8,94.9,102.0],[106.1,53.2,52.1],[135.6,123.4,112.2],[148.8,136.6,115.2],[51.1,39.9,25.7],[50.1,27.7,35.9],[82.7,30.8,36.9],[121.3,105.0,89.8],[44.0,23.6,24.7],[77.6,24.7,27.7],[69.4,42.0,29.8],[146.8,79.6,85.7],[19.6,18.6,20.6],[26.7,36.9,22.6],[30.8,35.9,40.9],[26.7,27.7,35.9],[50.1,69.4,57.2],[143.7,121.3,82.7],[243.4,238.3,190.5],[140.7,112.2,56.2],[168.1,137.6,85.7],[131.6,94.4,78.6],[220.8,171.2,148.6],[112.4,139.5,179.1],[100.0,123.7,77.4],[152.0,146.3,201.6],[118.1,215.2,193.7],[243.4,144.1,51.5],[92.1,104.5,189.2],[219.7,103.4,113.6],[107.9,69.5,123.7],[179.1,214.1,74.1],[254.7,185.8,53.7],[65.0,70.7,171.2],[80.8,168.9,84.2],[199.4,62.8,69.5],[255.0,226.5,36.8],[212.9,98.9,170.0],[10.7,152.0,183.6],[255.0,255.0,255.0],[227.6,227.6,227.6],[182.4,182.4,182.4],[139.5,139.5,138.4],[97.8,97.8,97.8],[60.5,60.5,60.5]], weights=[[-3.113,5.946,5.553],[-4.696,5.027,5.548],[-6.511,3.128,4.439],[-8.522,0.075,1.785],[-7.388,-0.625,0.224],[-1.276,-3.475,-2.601],[-3.859,-3.289,-2.533],[-4.149,-2.487,-1.702],[-7.053,-3.654,-3.593],[0.676,-1.125,-2.858],[-0.648,-1.101,-0.899],[-0.407,-1.26,1.715],[-0.681,0.625,0.516],[1.292,1.138,0.835],[0.073,2.603,1.882],[-2.577,-0.257,-0.005],[-2.573,-1.934,-1.373],[-2.621,1.197,2.089],[6.622,5.769,4.074],[-2.765,1.084,1.836],[3.413,-3.578,-2.219],[4.194,-0.329,-0.35],[-1.102,1.421,0.792],[2.905,-0.918,-0.569],[-0.866,-2.759,-2.098],[2.393,-0.533,-0.31],[0.063,-0.772,-0.956],[4.527,0.886,0.538],[3.984,2.065,2.142],[-2.734,-3.378,-2.691],[-3.05,-4.668,-1.53],[-3.785,-3.139,0.155],[0.931,-0.574,-0.076],[-1.202,-3.015,-1.356],[3.85,4.511,0.361],[-1.765,-2.322,-0.603],[-2.793,1.625,-0.357],[-5.074,2.185,-1.002],[1.384,-0.518,-1.009],[0.898,6.217,4.328],[5.287,2.355,2.519],[-0.164,0.934,1.301],[2.21,0.888,-0.358],[-0.086,3.536,3.192],[-6.493,-2.353,-1.95],[-6.044,-2.677,-1.588],[6.147,0.762,-7.115],[-1.474,3.791,2.18],[1.585,2.72,-1.662],[4.643,0.97,-6.224],[5.846,1.761,-4.55],[6.563,2.457,-1.651],[0.499,-1.373,-0.734],[5.242,-0.984,-3.975],[4.457,0.129,-1.125],[3.927,-0.293,-0.256],[4.844,1.787,0.99],[0.253,0.817,0.867],[7.913,2.963,0.448],[7.768,3.864,1.219],[-2.546,-0.875,-0.666],[-1.396,-1.446,-0.373],[-2.174,-0.402,0.608],[-2.175,-5.197,-2.044],[-3.908,-5.869,-1.828],[-3.799,-3.206,-2.791],[-4.261,-6.929,-2.572],[-1.705,-1.371,-2.826],[-2.101,-0.076,0.558],[-4.996,-1.139,-1.141],[3.193,-0.467,-3.049],[3.35,1.555,1.394],[-0.766,-0.178,-0.138],[-0.391,-0.462,-0.781],[4.108,0.824,0.618],[0.136,4.742,3.326],[3.974,1.176,1.759],[2.293,-1.252,-1.453],[0.182,-2.069,-1.449],[-3.67,-5.907,-2.42],[-11.114,0.897,3.624],[4.178,0.997,2.298],[-0.802,-3.361,-1.851],[-4.322,-2.84,-0.17],[-2.578,0.63,1.471],[-2.287,-3.096,-2.406],[-6.24,3.685,4.49],[-1.426,-3.898,-1.898],[-0.807,4.036,3.016],[-2.094,0.396,-0.182],[-10.915,-0.078,2.11],[-6.093,-3.064,3.603],[-2.385,-1.15,2.681],[-0.875,-0.247,0.721],[0.903,-0.622,-1.813],[-6.403,-2.487,-3.503],[-7.443,-2.365,-1.934],[6.304,2.186,1.621],[-8.802,-2.831,-3.226],[1.295,3.346,0.076],[8.936,4.438,-0.026],[2.122,1.675,1.04],[-0.182,1.347,0.26],[8.005,6.37,1.239],[-3.328,-3.35,-0.395],[-1.349,-3.978,-2.234],[-3.434,-4.924,-1.478],[-2.711,0.413,1.084],[-1.907,-1.55,-0.87],[8.981,8.352,2.95],[3.696,0.911,1.755],[5.276,1.854,1.027],[-2.821,0.348,0.496],[12.573,8.446,5.373],[-3.531,-4.121,-3.707],[-1.061,-1.911,-0.837],[3.138,2.324,0.147],[0.831,-1.653,-2.977],[9.949,3.793,-1.162],[-5.708,-1.861,-3.418],[2.022,2.444,-0.546],[9.144,4.368,-1.666],[8.061,4.241,-0.58],[-13.986,-10.604,2.559],[0.923,6.408,3.793],[-0.478,3.851,4.0],[-4.193,-3.174,-0.279],[-2.845,-0.561,0.525],[-5.134,-3.474,0.62],[-7.877,-2.434,-2.673],[11.337,-0.911,-0.503],[-1.883,3.376,0.121],[-4.231,2.684,3.549],[-5.737,-1.499,-1.0],[29.866,14.653,6.708],[5.845,4.739,2.465],[-8.502,4.248,4.946],[-7.095,2.733,-2.933],[33.124,-5.303,1.19],[10.849,36.945,18.239],[3.787,-4.696,-10.031],[-17.741,11.576,8.406],[4.129,-24.128,-22.206],[5.409,-11.32,18.888],[7.873,9.816,3.308],[9.036,-19.023,-30.687],[-13.58,10.962,36.91],[-40.504,-20.949,20.25],[-4.835,-25.145,-11.541],[25.122,15.535,-2.828],[18.424,-3.304,-26.163],[-15.895,-7.553,3.602],[1.324,-0.576,0.69],[-3.711,-1.435,-1.5],[4.008,0.059,0.207],[-3.562,0.686,0.087],[-1.326,-3.434,-1.629]]),
    "purple": dict(sigma=48.0, centers=[[61.0,106.0,212.0],[68.0,112.0,218.0],[75.0,121.0,239.0],[105.0,149.0,244.0],[86.0,133.0,248.0],[78.0,86.0,104.0],[112.0,144.0,207.0],[113.0,140.0,183.0],[113.0,117.0,135.0],[112.0,110.0,111.0],[180.0,178.0,176.0],[212.0,210.0,206.0],[182.0,182.0,197.0],[146.0,146.0,146.0],[201.0,195.0,189.0],[120.0,167.0,245.0],[168.0,164.0,158.0],[130.0,122.0,119.0],[228.0,226.0,221.0],[233.0,232.0,229.0],[226.0,221.0,215.0],[59.0,68.0,86.0],[50.0,58.0,76.0],[118.0,135.0,143.0],[90.0,99.0,115.0],[73.0,75.0,85.0],[42.0,46.0,54.0],[88.0,106.0,136.0],[103.0,91.0,90.0],[196.0,188.0,181.0],[110.0,97.0,95.0],[29.0,36.0,45.0],[24.0,29.0,36.0],[24.0,25.0,28.0],[69.5,135.1,228.6],[75.0,139.6,235.3],[88.4,150.7,244.2],[102.8,164.0,255.0],[112.9,170.7,255.0],[134.0,204.1,255.0],[152.9,205.2,255.0],[156.3,197.4,235.3],[203.0,223.0,233.0],[233.0,237.5,236.4],[135.1,188.5,241.9],[255.0,255.0,255.0],[177.4,181.9,188.5],[175.2,177.4,181.9],[121.8,174.1,234.2],[190.8,191.9,194.1],[125.1,125.1,125.1],[102.8,112.9,122.9],[125.1,159.6,197.4],[93.9,102.8,115.1],[164.0,229.7,255.0],[140.7,145.1,155.1],[88.4,88.4,88.4],[135.1,139.6,148.5],[197.4,240.8,255.0],[249.7,253.1,248.6],[233.0,248.6,253.1],[160.7,161.8,164.0],[193.0,184.1,174.1],[147.4,139.6,128.4],[69.5,79.5,90.6],[243.1,253.1,254.2],[59.4,60.6,50.5],[87.3,86.2,60.6],[156.3,150.7,128.4],[111.7,106.2,100.6],[125.1,122.9,92.8],[75.0,71.7,63.9],[27.2,30.5,21.6],[187.4,176.3,161.8],[70.6,73.9,49.4],[45.0,48.3,32.7],[148.5,141.8,104.0],[35.0,40.5,27.2],[155.1,146.2,105.1],[53.7,105.7,211.8],[82.0,135.0,241.1],[66.2,121.5,231.0],[53.7,107.9,220.8],[67.3,114.7,207.3],[87.6,127.1,203.9],[126.0,132.8,150.8],[68.4,102.3,165.5],[88.7,117.0,166.6],[164.4,165.5,166.6],[168.9,177.9,190.3],[199.4,199.4,200.5],[114.7,142.9,199.4],[130.5,156.5,206.2],[120.3,121.5,129.4],[87.6,88.7,96.6],[96.6,102.3,117.0],[65.0,86.5,124.9],[70.7,77.4,92.1],[149.7,142.9,135.0],[65.0,69.5,79.7],[237.8,234.4,233.2],[106.8,112.4,129.4],[212.9,215.2,220.8],[255.0,250.2,244.5],[175.7,185.8,194.9],[220.8,216.3,211.8],[227.6,220.8,215.2],[184.7,179.1,172.3],[102.3,150.8,236.6],[117.0,165.5,234.4],[189.2,187.0,177.9],[83.1,106.8,153.1],[79.7,76.3,72.9],[57.1,58.3,61.6],[102.3,162.1,255.0],[179.1,184.7,179.1],[86.5,142.9,255.0],[76.3,71.8,68.4],[35.7,41.3,52.6],[28.9,33.4,44.7],[155.3,150.8,141.8],[75.2,74.1,76.3],[19.9,25.5,35.7],[111.3,109.1,115.8],[113.6,174.5,255.0],[161.7,116.8,97.7],[255.0,209.4,182.2],[138.6,171.3,218.9],[123.6,152.2,96.3],[186.2,179.4,246.2],[145.4,255.0,236.6],[255.0,176.7,65.0],[114.0,129.0,231.2],[255.0,127.7,139.9],[133.1,86.8,152.2],[218.9,255.0,92.3],[255.0,227.1,67.7],[81.4,88.2,209.4],[100.4,206.7,104.5],[243.4,78.6,86.8],[255.0,255.0,47.3],[255.0,122.2,208.0],[15.8,186.2,224.4],[255.0,255.0,255.0],[255.0,255.0,255.0],[223.0,223.0,223.0],[171.3,171.3,169.9],[120.9,120.9,120.9],[75.9,75.9,75.9]], weights=[[-12.209,-4.831,-1.006],[-8.616,-3.257,-0.774],[-1.209,-2.305,-2.847],[-1.715,3.831,2.812],[3.044,0.381,-1.216],[1.138,-0.881,-2.142],[-2.916,5.314,3.643],[4.539,1.124,0.263],[3.09,-0.088,-0.951],[-2.631,1.132,0.769],[-0.693,0.146,0.557],[0.318,-1.053,-1.423],[1.009,-1.689,-0.867],[2.449,0.008,-0.455],[3.045,-2.172,-1.408],[-4.275,0.241,1.476],[0.562,3.352,2.888],[-2.143,-0.554,-0.559],[-5.607,1.209,-0.074],[-6.272,1.042,-0.075],[-3.632,1.093,-0.064],[-0.781,-2.929,-3.496],[-0.363,-2.087,-2.78],[3.849,-2.118,-1.826],[2.083,1.196,-0.232],[-1.33,-2.092,-2.15],[1.711,0.831,0.217],[5.96,1.378,0.409],[-2.654,0.528,0.309],[2.476,-0.88,0.014],[-2.984,1.046,0.574],[0.702,-0.174,-0.909],[-0.342,-0.885,-1.232],[-0.706,-0.987,-0.922],[17.126,-4.05,-4.689],[16.212,-3.6,-4.66],[11.04,-2.65,-3.32],[3.857,-3.433,-2.014],[0.121,-3.124,-0.854],[0.865,-3.026,-1.513],[5.777,1.777,3.082],[0.91,-1.275,-0.01],[-2.925,3.054,1.525],[-6.007,0.738,-0.187],[-3.219,-3.665,-1.54],[7.793,-2.04,-0.75],[-1.476,-2.446,-1.782],[-1.836,-0.973,-0.623],[-5.37,-3.177,-2.078],[1.553,-3.331,-2.43],[-1.226,-0.795,-0.564],[1.097,1.068,0.451],[-3.541,-0.999,-2.231],[1.289,1.409,0.243],[3.396,6.503,2.952],[2.777,-0.474,-1.353],[-2.402,-0.4,-0.222],[3.239,-1.261,-1.798],[-1.596,6.484,3.616],[3.608,-1.842,-1.079],[-0.817,0.363,0.181],[-0.418,1.88,1.133],[1.778,0.634,1.449],[0.73,-0.346,-0.029],[-0.809,-2.206,-2.594],[3.157,-1.106,-0.506],[1.992,2.567,3.359],[-2.959,0.736,3.226],[1.161,1.355,1.988],[-4.247,1.248,1.348],[-8.681,-0.285,2.861],[-1.262,0.221,1.591],[0.285,0.05,0.397],[1.204,3.863,4.322],[0.547,2.533,4.44],[3.839,3.627,3.933],[-4.088,-1.433,1.321],[2.414,2.011,2.201],[-2.259,-0.852,1.466],[-8.186,-6.277,-2.61],[7.666,-1.2,-2.567],[5.083,-5.019,-4.794],[-5.451,-7.295,-4.608],[-2.419,-2.686,0.194],[2.216,2.609,3.329],[4.472,-1.386,-2.046],[-0.24,0.299,2.909],[6.681,0.786,2.082],[-1.037,1.792,1.233],[-3.493,-2.317,-1.946],[2.377,-2.857,-2.311],[-1.98,4.56,2.987],[-8.284,3.136,1.616],[0.654,-0.541,-0.913],[-0.894,-0.269,-0.925],[1.588,1.425,0.05],[3.804,-0.386,-1.78],[-0.564,-2.304,-2.873],[1.935,0.263,0.194],[-1.181,-2.146,-2.129],[-5.195,0.681,-0.123],[2.588,0.706,-0.3],[-1.803,1.243,0.351],[3.754,-2.034,-1.231],[-1.885,-3.854,-3.032],[-1.935,0.593,-0.342],[-3.376,1.296,0.146],[0.287,1.055,1.576],[1.02,2.149,1.106],[-4.556,-0.452,-0.126],[0.552,-1.229,-0.452],[5.347,1.145,1.712],[-2.213,-0.807,0.245],[0.718,0.499,0.703],[3.962,-2.935,-1.748],[-1.778,-2.14,-1.594],[8.936,-2.247,-3.218],[-1.605,-0.421,0.707],[1.258,0.212,-0.652],[0.505,-0.354,-1.086],[2.288,1.661,1.383],[-1.841,-1.361,-0.725],[-1.232,-1.667,-1.987],[-0.99,1.086,0.205],[0.164,-4.188,-1.588],[2.011,-2.223,-4.843],[14.039,8.917,9.132],[-9.025,-0.461,-0.276],[-10.794,-3.007,3.269],[17.995,12.44,13.918],[-2.051,10.227,-0.239],[-8.649,-2.684,2.337],[-20.472,16.798,13.902],[-2.627,7.917,12.05],[3.145,-0.444,-3.349],[-11.998,11.14,-2.072],[2.388,-1.569,1.676],[-49.311,2.357,8.542],[-0.381,-1.692,-5.686],[2.913,-3.164,-14.895],[7.828,-16.257,-13.746],[3.677,26.395,10.314],[-3.017,21.695,20.412],[7.793,-2.04,-0.75],[7.793,-2.04,-0.75],[-5.024,1.53,0.26],[-1.327,1.426,1.275],[-2.036,-0.189,-0.038],[-2.023,-1.243,-0.486]]),
}


@dataclass(frozen=True)
class Film:
    name: str
    kind: str                     # "matrix" | "natural"
    params: tuple = ()            # modelli misurati
    ev_bias: float = 0.0          # esposizione di riferimento delle scene
    # profili statistici
    tint: tuple = ()              # 8 fasce tonali × (ΔR, ΔG, ΔB) in 0–255
    black: float = 0.0            # nero della scansione (0–255)
    white: float = 255.0          # bianco della scansione
    contrast: float = 0.0         # S-curve
    sat: float = 1.0
    # comportamento fisico
    grain: float = 9.0            # grana predefinita
    grain_size: float = 1.0       # dimensione relativa del grano
    halation: float = 0.25
    halo_tint: tuple = (1.0, 0.55, 0.35)
    # risposta all'ISO (solo LomoChrome XR 100–400)
    iso_ev: float = 0.0
    iso_warm: tuple = ()          # (boost croma, calo luminanza) dei toni caldi a ISO 400
    iso_contrast: float = 0.0
    resid: str = ""               # chiave in RESIDUAL
    desc: str = ""


FILMS = {
    "LomoChrome Purple": Film(
        "LomoChrome Purple", "matrix", tuple(MEASURED["purple"]), resid="purple",
        grain=9, grain_size=1.1, halation=0.25, halo_tint=(1.0, 0.45, 0.85), iso_ev=-0.3,
        desc="Verdi e gialli → viola e rosa, blu → verde acqua. Contrasto morbido."),
    "LomoChrome Turquoise": Film(
        "LomoChrome Turquoise", "matrix", tuple(MEASURED["turquoise"]), resid="turquoise",
        grain=10, grain_size=1.1, halation=0.25, halo_tint=(1.0, 0.85, 0.6),
        iso_ev=-1.200, iso_warm=(0.094, 0.220),     # tarato sul cielo reale a ISO 400
        desc="Cielo blu → oro/pesca, toni caldi e verdi → turchese e smeraldo."),
    "LomoChrome Metropolis": Film(
        "LomoChrome Metropolis", "matrix", tuple(MEASURED["metropolis"]),
        grain=9, grain_size=1.0, halation=0.2, halo_tint=(1.0, 0.8, 0.6), iso_ev=-0.3, iso_contrast=1.5,
        desc="Colori smorzati, contrasto deciso, toni freddi e metallici."),
    "LomoChrome Color '92": Film(
        "LomoChrome Color '92", "natural",
        tint=((0, 1, -7), (1, 3, -10), (7, 5, -15), (8, 1, -17),
              (15, -2, -23), (15, -1, -17), (15, -1, -10), (7, -2, -5)),
        black=21, white=221, contrast=0.0, sat=0.82,
        grain=16, grain_size=1.45, halation=0.35, halo_tint=(1.0, 0.5, 0.3),
        desc="Pastello anni '90, neri sollevati, luci compresse, grana evidente."),
    "LomoChrome Color '92 Sun-kissed": Film(
        "LomoChrome Color '92 Sun-kissed", "natural",
        tint=((2, 0, -6), (3, 1, -12), (7, 1, -22), (8, 1, -24),
              (10, 0, -27), (11, -1, -21), (8, -2, -10), (4, -1, -3)),
        black=12, white=238, contrast=2.5, sat=0.9,
        grain=13, grain_size=1.3, halation=0.35, halo_tint=(1.0, 0.55, 0.25),
        desc="Dorata, gialli e arancio caldi, più contrastata della Color '92."),
    "LomoChrome Classicolor": Film(
        "LomoChrome Classicolor", "natural",
        tint=((3, 0, -4), (9, -2, -8), (11, -2, -14), (9, -1, -11),
              (9, 0, -14), (13, -1, -20), (13, -1, -15), (4, 0, -8)),
        black=21, white=229, contrast=1.0, sat=1.05,
        grain=7, grain_size=0.9, halation=0.2, halo_tint=(1.0, 0.5, 0.3),
        desc="Colori naturali e vividi, pelle calda, rossi pieni, ombre neutre."),
}

ASPECTS = {
    "Originale": None,
    "35 mm (3:2)": 3 / 2,
    "6×4,5 (4:3)": 4 / 3,
    "6×6 (1:1)": 1.0,
    "6×7 (7:6)": 7 / 6,
    "Mezzo formato (3:4)": 3 / 4,
    "Panoramico (65:24)": 65 / 24,
}
MAX_ZOOM = 8.0


# ═══════════════════════════════════════════════════════════════
# UTILITÀ NUMERICHE
# ═══════════════════════════════════════════════════════════════

def _box1d(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim; pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    return (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
            - np.take(c, np.arange(0, n), axis=axis)) / np.float32(2 * r + 1)


def _gauss(a, sigma):
    a = a.astype(np.float32, copy=False)
    if sigma < 0.3:
        return a
    if sigma < 1.3:
        xs = np.arange(-2, 3); k = np.exp(-(xs ** 2) / (2 * sigma * sigma)); k /= k.sum()
        for axis in (0, 1):
            pad = [(0, 0)] * a.ndim; pad[axis] = (2, 2)
            p = np.pad(a, pad, mode="edge"); n = a.shape[axis]
            a = sum(np.float32(k[i]) * np.take(p, np.arange(i, i + n), axis=axis) for i in range(5))
        return a
    r = max(1, int(round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    for _ in range(3):
        a = _box1d(_box1d(a, r, 0), r, 1)
    return a


def _lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _radial(h, w):
    y = (np.arange(h, dtype=np.float32) + 0.5) / h * 2 - 1
    x = (np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1
    return np.sqrt(x[None, :] ** 2 + y[:, None] ** 2) / math.sqrt(2)


def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _scurve(x, k):
    if k < 1e-3:
        return x
    s = lambda t: 1 / (1 + np.exp(-t))
    a, b = s(-k / 2), s(k / 2)
    return ((s(k * (x - 0.5)) - a) / (b - a)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# TRASFORMAZIONI COLORE
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=16)
def _curve_tables(params):
    """Curve monotone per canale campionate a 1024 punti."""
    p = np.array(params, float)
    xs = np.linspace(0, 1, K_CURVE)
    t = np.linspace(0, 1, 1024)
    tabs = []
    for c in range(3):
        k = p[12 + c * K_CURVE: 12 + (c + 1) * K_CURVE]
        y = np.concatenate([[k[0]], k[0] + np.cumsum(np.exp(k[1:]))])
        tabs.append(np.clip(np.interp(t, xs, y), 0, 1))
    return np.stack(tabs, 1).astype(np.float32)


def film_matrix(arr, film, ev):
    """Modello misurato: luce lineare × esposizione → scambio canali → curve."""
    p = np.array(film.params, np.float32)
    M = p[:9].reshape(3, 3); b = p[9:12]
    L = _srgb_to_lin(arr) * np.float32(2.0 ** ev)
    L = np.clip(L @ M.T + b, 0, 1) ** (1 / 2.2)
    idx = np.clip(L * 1023 + 0.5, 0, 1023).astype(np.int32)
    tab = _curve_tables(film.params)
    out = np.stack([tab[idx[..., c], c] for c in range(3)], -1)
    if film.resid:
        lut = _resid_lut(film.resid)
        xe = _lin_to_srgb(np.clip(_srgb_to_lin(arr) * np.float32(2.0 ** ev), 0, 1))
        out = out + _trilinear(lut, xe) / 255.0
    return np.clip(out, 0, 1)


def _lin_to_srgb(l):
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(l, 1 / 2.4) - 0.055).astype(np.float32)


@lru_cache(maxsize=8)
def _resid_lut(key, n=25):
    """LUT 3D n³ del residuo RBF, calcolata una volta sola."""
    R = RESIDUAL[key]
    C = np.array(R["centers"], np.float32); Wt = np.array(R["weights"], np.float32)
    g = np.linspace(0, 255, n, dtype=np.float32)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    out = np.zeros((G.shape[0], 3), np.float32)
    for i in range(0, G.shape[0], 4096):
        d2 = ((G[i:i + 4096, None, :] - C[None]) ** 2).sum(-1)
        out[i:i + 4096] = np.exp(-d2 / (2 * R["sigma"] ** 2)) @ Wt
    return np.clip(out, -150, 150).reshape(n, n, n, 3)


def _trilinear(lut, x):
    n = lut.shape[0]
    f = np.clip(x, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(f).astype(np.int32), 0, n - 2); t = (f - i0).astype(np.float32)
    r, g, b = i0[..., 0], i0[..., 1], i0[..., 2]
    tr, tg, tb = t[..., 0:1], t[..., 1:2], t[..., 2:3]
    c00 = lut[r, g, b] * (1 - tr) + lut[r + 1, g, b] * tr
    c01 = lut[r, g, b + 1] * (1 - tr) + lut[r + 1, g, b + 1] * tr
    c10 = lut[r, g + 1, b] * (1 - tr) + lut[r + 1, g + 1, b] * tr
    c11 = lut[r, g + 1, b + 1] * (1 - tr) + lut[r + 1, g + 1, b + 1] * tr
    c0 = c00 * (1 - tg) + c10 * tg; c1 = c01 * (1 - tg) + c11 * tg
    return c0 * (1 - tb) + c1 * tb


def film_natural(arr, film, ev):
    """Impronta statistica: esposizione, saturazione, curva, dominante per fascia tonale."""
    a = np.clip(arr * np.float32(2.0 ** (ev * 0.6)), 0, 1)
    L = _lum(a)[..., None]
    a = np.clip(L + (a - L) * film.sat, 0, 1)
    a = _scurve(a, film.contrast)
    bk, wh = film.black / 255, film.white / 255
    a = bk + (wh - bk) * a
    # dominante interpolata sulle 8 fasce (centri 16, 48, …, 240)
    centers = (np.arange(8) * 32 + 16) / 255
    t = np.array(film.tint, np.float32) / 255 * 0.75
    Ln = _lum(a)
    off = np.stack([np.interp(Ln, centers, t[:, c]) for c in range(3)], -1)
    return np.clip(a + off, 0, 1).astype(np.float32)


def _warm_boost(o, kc, kl):
    """Saturazione selettiva dei toni pesca/arancio (Turquoise a ISO 400)."""
    Lm = _lum(o)[..., None]
    b = (o[..., 0] + o[..., 1]) / 2 - o[..., 2]; a = o[..., 0] - o[..., 1]
    # solo pesca/arancio: giallo-caldo dominante (b) e rosso moderato, esclusi rosa e magenta
    w = (np.clip((b - 0.03) / 0.10, 0, 1) * np.clip((a + 0.02) / 0.08, 0, 1)
         * np.clip((b - 0.55 * a) / 0.06, 0, 1))
    w = (w * w * (3 - 2 * w))[..., None]
    out = Lm * (1 - kl * w) + (o - Lm) * (1 + kc * w)
    return np.clip(out, 0, 1)


def apply_film(arr, film, ev, strength, iso=200):
    if film.kind == "matrix":
        t = float(np.clip(math.log2(max(iso, 50) / 200.0), 0, 1))   # 100–200 → 0, 400 → 1
        out = film_matrix(arr, film, ev + film.iso_ev * t)
        if t > 0 and film.iso_warm:
            out = _warm_boost(out, film.iso_warm[0] * t, film.iso_warm[1] * t)
        if t > 0 and film.iso_contrast:
            out = _scurve(out, film.iso_contrast * t)
    else:
        out = film_natural(arr, film, ev)
    return out * strength + arr * (1 - strength)


# ═══════════════════════════════════════════════════════════════
# SVILUPPO
# ═══════════════════════════════════════════════════════════════

def _grain(arr, ppmm, rng, amount, size, chroma=0.35):
    h, w = arr.shape[:2]
    sig = 0.012 * ppmm * size
    n = rng.standard_normal((h, w), dtype=np.float32)
    if sig >= 0.3:
        n = _gauss(n, sig); n /= n.std() + 1e-6
    c = rng.standard_normal((h, w, 3), dtype=np.float32)
    if sig * 1.6 >= 0.3:
        c = _gauss(c, sig * 1.6); c /= c.std() + 1e-6
    L = _lum(arr)
    wgt = (0.35 + 2.6 * L * (1 - L))[..., None]       # più grana nei mezzitoni
    amp = amount / 255.0
    return arr + (n[..., None] * (1 - chroma) + c * chroma) * wgt * amp


def develop(arr, s, ppmm, seed, fast=False):
    film = FILMS[s["film"]]
    h, w = arr.shape[:2]
    R = lambda k: np.random.default_rng([seed & 0xFFFFFFFF, k])

    if not fast and s["softness"] > 0:
        soft = s["softness"]
        arr = _gauss(arr, 0.02 * ppmm * soft)
        blurred = _gauss(arr, 0.08 * ppmm * soft)
        m = (np.clip(_radial(h, w) ** 2 * 1.3, 0, 1) * soft)[..., None]
        arr = arr + (blurred - arr) * m

    arr = apply_film(np.clip(arr, 0, 1), film, s["ev"], s["strength"], s.get("iso", 200))

    if s["halation"] > 0 and not fast:
        hi = np.clip((_lum(arr) - 0.72) / 0.28, 0, 1) ** 2
        glow = _gauss(hi, 0.35 * ppmm)
        arr = arr + glow[..., None] * np.array(film.halo_tint, np.float32) * (0.5 * s["halation"])
        arr = np.where(arr > 0.9, 0.9 + 0.1 * np.tanh((arr - 0.9) / 0.1), arr)

    if s["vignette"] > 0:
        t = np.clip((_radial(h, w) - 0.3) / 0.75, 0, 1)
        arr = arr * (1 - s["vignette"] * t * t * (3 - 2 * t))[..., None]

    if s["grain"] > 0:
        arr = _grain(arr, ppmm, R(4), s["grain"], film.grain_size)
    return np.clip(arr, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# INQUADRATURA (zoom / pan sempre dentro il fotogramma)
# ═══════════════════════════════════════════════════════════════

@dataclass
class View:
    zoom: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    rot: int = 0
    seed: int = 0


def base_window(W, H, aspect):
    return (H * aspect, H) if W / H > aspect else (W, W / aspect)


def clamp_view(W, H, aspect, v):
    v.zoom = min(max(v.zoom, 1.0), MAX_ZOOM)
    bw, bh = base_window(W, H, aspect)
    hx, hy = bw / v.zoom / 2 / W, bh / v.zoom / 2 / H
    v.cx = min(max(v.cx, hx), 1 - hx)
    v.cy = min(max(v.cy, hy), 1 - hy)
    return v


def crop_box(size, aspect, v):
    W, H = size
    clamp_view(W, H, aspect, v)
    bw, bh = base_window(W, H, aspect)
    cw, ch = bw / v.zoom, bh / v.zoom
    x0, y0 = v.cx * W - cw / 2, v.cy * H - ch / 2
    return (x0, y0, x0 + cw, y0 + ch)


def frame_aspect(s, size):
    a = ASPECTS[s["aspect"]]
    if a is None:
        return size[0] / size[1]
    if s.get("auto_orient", True) and (size[0] < size[1]) != (a < 1) and abs(a - 1) > 1e-6:
        a = 1 / a          # segue l'orientamento della foto
    return a


def oriented(img, rot):
    return {0: img, 90: img.transpose(Image.Transpose.ROTATE_270),
            180: img.transpose(Image.Transpose.ROTATE_180),
            270: img.transpose(Image.Transpose.ROTATE_90)}[rot % 360]


def load_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def render(src, s, view, out_long=3000, fast=False):
    aspect = frame_aspect(s, src.size)
    box = crop_box(src.size, aspect, view)
    if aspect >= 1:
        ow, oh = int(out_long), max(1, int(round(out_long / aspect)))
    else:
        ow, oh = max(1, int(round(out_long * aspect))), int(out_long)
    # non ingrandire oltre la risoluzione disponibile
    scale = min(1.0, (box[2] - box[0]) / ow) if not fast else 1.0
    if scale < 1.0:
        ow, oh = max(1, int(ow * scale)), max(1, int(oh * scale))
    if fast:
        img = src.resize((ow, oh), Image.BILINEAR, box=box)
    else:
        img = src.resize((ow, oh), Image.LANCZOS, box=box, reducing_gap=3.0)
    arr = np.asarray(img, np.float32) / 255
    ppmm = max(ow, oh) / 36.0            # riferimento: lato lungo = 36 mm (35 mm)
    arr = develop(arr, s, ppmm, view.seed, fast)
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8))


def save_image(img, path):
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, "JPEG", quality=96, subsampling=0)
    elif ext in (".tif", ".tiff"):
        img.save(path, "TIFF", compression="tiff_lzw")
    else:
        img.save(path, "PNG")


# ═══════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════

class ExportWorker(QThread):
    progress = pyqtSignal(str, int)
    done = pyqtSignal(int, list)

    def __init__(self, jobs, settings, out_long):
        super().__init__()
        self.jobs, self.s, self.out_long = jobs, settings, out_long

    def run(self):
        errs, n = [], len(self.jobs)
        for i, (path, view, out) in enumerate(self.jobs):
            self.progress.emit(f"[{i+1}/{n}] {Path(path).name}…", int(i / n * 100))
            try:
                src = oriented(load_image(path), view.rot)
                save_image(render(src, self.s, view, self.out_long), out)
            except Exception as e:
                errs.append(f"{Path(path).name}: {e}")
        self.progress.emit("Esportazione completata", 100)
        self.done.emit(n, errs)


# ═══════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════

PREVIEW_FULL = 1200
PREVIEW_FAST = 700
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
PAL = [(110, 60, 160), (150, 80, 190), (60, 170, 170), (230, 190, 120),
       (90, 100, 110), (230, 160, 90), (200, 70, 60)]


def pil2pm(pil):
    i = pil.convert("RGBA")
    return QPixmap.fromImage(QImage(i.tobytes("raw", "RGBA"), i.width, i.height,
                                    i.width * 4, QImage.Format.Format_RGBA8888).copy())


def _drop_files(e):
    return [u.toLocalFile() for u in e.mimeData().urls()
            if u.toLocalFile().lower().endswith(IMG_EXTS)]


class OrigView(QWidget):
    dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._pm = None; self._box = None
        self.setAcceptDrops(True); self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, pm, box):
        self._pm, self._box = pm, box; self.update()

    def clear(self):
        self._pm = None; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if not self._pm:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "📷\nTrascina immagini qui\no usa 📂 Apri")
            return
        pw, ph = self._pm.width(), self._pm.height()
        s = min(self.width() / pw, self.height() / ph) * 0.96
        r = QRectF((self.width() - pw * s) / 2, (self.height() - ph * s) / 2, pw * s, ph * s)
        p.drawPixmap(r, self._pm, QRectF(0, 0, pw, ph))
        if self._box:
            x0, y0, x1, y1 = self._box
            b = QRectF(r.x() + x0 * r.width(), r.y() + y0 * r.height(),
                       (x1 - x0) * r.width(), (y1 - y0) * r.height())
            dim = QColor(0, 0, 0, 120)
            p.fillRect(QRectF(r.x(), r.y(), r.width(), b.y() - r.y()), dim)
            p.fillRect(QRectF(r.x(), b.bottom(), r.width(), r.bottom() - b.bottom()), dim)
            p.fillRect(QRectF(r.x(), b.y(), b.x() - r.x(), b.height()), dim)
            p.fillRect(QRectF(b.right(), b.y(), r.right() - b.right(), b.height()), dim)
            p.setPen(QPen(QColor(150, 80, 190), 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(b)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        f = _drop_files(e)
        if f: self.dropped.emit(f)


class FilmView(QWidget):
    """Anteprima interattiva: rotella = zoom, trascina = sposta, doppio clic = reimposta."""
    panned = pyqtSignal(float, float)
    zoomed = pyqtSignal(float, float, float)
    reset = pyqtSignal()
    interact = pyqtSignal(bool)
    dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._pm = None; self._last = None
        self.setAcceptDrops(True); self.setMouseTracking(True)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_result(self, pm):
        self._pm = pm; self.update()

    def clear(self):
        self._pm = None; self.update()

    def _rect(self):
        pw, ph = self._pm.width(), self._pm.height()
        s = min(self.width() / pw, (self.height() - 22) / ph) * 0.97
        return QRectF((self.width() - pw * s) / 2, (self.height() - 22 - ph * s) / 2, pw * s, ph * s)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if not self._pm:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Il risultato\napparirà qui")
            return
        p.drawPixmap(self._rect(), self._pm, QRectF(0, 0, self._pm.width(), self._pm.height()))
        p.setPen(QColor(140, 140, 140))
        p.drawText(QRectF(0, self.height() - 20, self.width(), 18), Qt.AlignmentFlag.AlignCenter,
                   "Rotella: zoom  ·  Trascina: sposta  ·  Doppio clic: reimposta")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pm:
            self._last = e.position(); self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.interact.emit(True)

    def mouseMoveEvent(self, e):
        if self._last is not None:
            d = e.position() - self._last; self._last = e.position()
            r = self._rect(); self.panned.emit(d.x() / r.width(), d.y() / r.height())
        elif self._pm:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._rect().contains(e.position())
                           else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._last is not None:
            self._last = None; self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.interact.emit(False)

    def mouseDoubleClickEvent(self, e):
        if self._pm: self.reset.emit()

    def wheelEvent(self, e):
        if not self._pm: return
        r = self._rect(); pos = e.position()
        u = min(max((pos.x() - r.x()) / r.width(), 0), 1)
        v = min(max((pos.y() - r.y()) / r.height(), 0), 1)
        steps = e.angleDelta().y() / 120.0
        if steps: self.zoomed.emit(u, v, 1.12 ** steps)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        f = _drop_files(e)
        if f: self.dropped.emit(f)


class SldRow(QWidget):
    changed = pyqtSignal(int)

    def __init__(self, label, lo, hi, default, sfx="%", fmt=None):
        super().__init__(); self.sfx = sfx; self.fmt = fmt
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1)
        lb = QLabel(label); lb.setFixedWidth(104); lay.addWidget(lb)
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(lo, hi); self.sl.setValue(default)
        lay.addWidget(self.sl, 1)
        self.vl = QLabel(); self.vl.setFixedWidth(56)
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._sh(default); lay.addWidget(self.vl)
        self.sl.valueChanged.connect(lambda v: (self._sh(v), self.changed.emit(v)))

    def _sh(self, v): self.vl.setText(self.fmt(v) if self.fmt else f"{v}{self.sfx}")
    def value(self): return self.sl.value()

    def set_silent(self, v):
        self.sl.blockSignals(True); self.sl.setValue(v); self._sh(v); self.sl.blockSignals(False)


SS = """
QMainWindow{background:#1c1b20}
QWidget{color:#ddd;font-size:12px}
QScrollArea{border:none;background:#1c1b20}
#panel{background:#1c1b20}
QGroupBox{border:1px solid #45404f;border-radius:6px;margin-top:12px;padding:12px 8px 8px;font-weight:bold;color:#b0a8c0}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px}
QCheckBox::indicator{width:15px;height:15px;border:1px solid #666;border-radius:3px;background:#2a2830}
QCheckBox::indicator:checked{background:#7a45a8;border-color:#7a45a8}
QSlider::groove:horizontal{background:#3a3640;height:5px;border-radius:2px}
QSlider::handle:horizontal{background:#a56fd6;width:14px;height:14px;margin:-5px 0;border-radius:7px}
QSlider::sub-page:horizontal{background:#7a45a8;border-radius:2px}
QComboBox{background:#2a2830;border:1px solid #555;border-radius:4px;padding:4px 8px;color:#ddd}
QComboBox QAbstractItemView{background:#2a2830;color:#ddd;selection-background-color:#7a45a8}
QProgressBar{background:#2a2830;border:none;border-radius:3px;height:8px}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6e3ca0,stop:.5 #3caaaa,stop:1 #e6be78);border-radius:3px}
QStatusBar{background:#141318;color:#888}
QMenuBar{background:#18171c;color:#bbb}
QMenuBar::item:selected{background:#7a45a8}
QMenu{background:#2a2830;color:#ddd;border:1px solid #444}
QMenu::item:selected{background:#7a45a8}
QListWidget{background:#2a2830;border:1px solid #444;border-radius:4px;color:#ccc;font-size:11px}
QListWidget::item{padding:3px 6px}
QListWidget::item:selected{background:#7a45a8;color:white}
"""
BG = ("QPushButton{background:#7a45a8;color:white;border:none;border-radius:6px;"
      "padding:9px 14px;font-size:13px;font-weight:bold}"
      "QPushButton:hover{background:#8d55bd}QPushButton:pressed{background:#5e3383}"
      "QPushButton:disabled{background:#333;color:#666}")
BF = ("QPushButton{background:#2a2830;color:#ccc;border:1px solid #555;"
      "border-radius:6px;padding:6px 10px;font-size:12px}"
      "QPushButton:hover{border-color:#a56fd6;background:#34313b}"
      "QPushButton:disabled{color:#555;border-color:#3a3a3a}")
BR = ("QPushButton{background:#6b2b2b;color:#ccc;border:1px solid #855;border-radius:6px;"
      "padding:6px 10px;font-size:11px}QPushButton:hover{background:#7b3535}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LomoChrome Simulator v1")
        self.setMinimumSize(1100, 720); self.resize(1420, 860)
        self.setStyleSheet(SS)
        self.items = []; self.cur = -1; self.wk = None
        self._dragging = False; self._want_fast = False
        self.rt = QTimer(self); self.rt.setSingleShot(True); self.rt.timeout.connect(self._render_preview)
        self.ft = QTimer(self); self.ft.setSingleShot(True); self.ft.timeout.connect(self._settle)
        self._menu(); self._ui()
        self.statusBar().showMessage("Trascina immagini o premi 📂 Apri")

    def _menu(self):
        mb = self.menuBar(); fm = mb.addMenu("&File")
        for txt, sc, fn in (("&Apri…", "Ctrl+O", self.open_files),
                            ("&Salva questa…", "Ctrl+S", self.save_current),
                            ("&Esporta tutte…", "Ctrl+Shift+S", self.export_all)):
            a = QAction(txt, self); a.setShortcut(sc); a.triggered.connect(fn); fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Esci", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close); fm.addAction(a)
        vm = mb.addMenu("&Inquadratura")
        for txt, sc, fn in (("Zoom +", "Ctrl++", lambda: self._on_zoom(0.5, 0.5, 1.15)),
                            ("Zoom −", "Ctrl+-", lambda: self._on_zoom(0.5, 0.5, 1 / 1.15)),
                            ("Reimposta", "Ctrl+0", self.reset_view),
                            ("Ruota a destra", "Ctrl+R", lambda: self.rotate(90)),
                            ("Ruota a sinistra", "Ctrl+Shift+R", lambda: self.rotate(-90))):
            a = QAction(txt, self); a.setShortcut(sc); a.triggered.connect(fn); vm.addAction(a)

    def _ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setContentsMargins(6, 6, 6, 4); root.setSpacing(4)

        bar = QLabel(); bar.setFixedHeight(26)
        pm = QPixmap(1200, 26); p = QPainter(pm); bw = 1200 / len(PAL)
        for i, c in enumerate(PAL): p.fillRect(int(i * bw), 0, int(bw) + 2, 26, QColor(*c))
        p.setPen(QColor(255, 255, 255, 220)); f = p.font(); f.setPointSize(7); f.setBold(True); p.setFont(f)
        p.drawText(8, 18, "LOMOCHROME SIMULATOR · PURPLE · TURQUOISE · METROPOLIS · COLOR '92 · SUN-KISSED · CLASSICOLOR")
        p.end(); bar.setPixmap(pm); bar.setScaledContents(True); root.addWidget(bar)

        sp = QSplitter(Qt.Orientation.Horizontal)
        inner = QWidget(); inner.setObjectName("panel")
        cl = QVBoxLayout(inner); cl.setContentsMargins(4, 4, 8, 4); cl.setSpacing(4)

        g = QGroupBox("📂 File"); gl = QVBoxLayout(g); rw = QHBoxLayout()
        b = QPushButton("📂 Apri"); b.setStyleSheet(BF); b.clicked.connect(self.open_files); rw.addWidget(b)
        b = QPushButton("✕"); b.setStyleSheet(BR); b.setFixedWidth(32); b.clicked.connect(self.clear_list); rw.addWidget(b)
        gl.addLayout(rw)
        self.flist = QListWidget(); self.flist.setMaximumHeight(96)
        self.flist.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.flist.currentRowChanged.connect(self._on_select); gl.addWidget(self.flist)
        cl.addWidget(g)

        rw = QHBoxLayout()
        self.bsave = QPushButton("💾 Salva"); self.bsave.setStyleSheet(BG); self.bsave.clicked.connect(self.save_current)
        self.ball = QPushButton("📦 Esporta tutte"); self.ball.setStyleSheet(BG); self.ball.clicked.connect(self.export_all)
        rw.addWidget(self.bsave); rw.addWidget(self.ball); cl.addLayout(rw)

        g = QGroupBox("🎞️ Pellicola"); gl = QVBoxLayout(g)
        self.cfilm = QComboBox(); self.cfilm.addItems(list(FILMS)); gl.addWidget(self.cfilm)
        self.ldesc = QLabel(); self.ldesc.setWordWrap(True)
        self.ldesc.setStyleSheet("color:#9a92a8;font-size:11px"); gl.addWidget(self.ldesc)
        rw = QHBoxLayout(); rw.addWidget(QLabel("ISO (EI)"))
        self.ciso = QComboBox(); self.ciso.addItems(["100", "200", "400"]); self.ciso.setCurrentText("200")
        rw.addWidget(self.ciso, 1); gl.addLayout(rw)
        self.sev = SldRow("Esposizione", -20, 20, 0, fmt=lambda v: f"{v/10:+.1f} EV"); gl.addWidget(self.sev)
        self.si = SldRow("Intensità", 0, 100, 100); gl.addWidget(self.si)
        cl.addWidget(g)

        g = QGroupBox("🌫️ Grana e ottica"); gl = QVBoxLayout(g)
        self.sg = SldRow("Grana", 0, 40, 9, ""); gl.addWidget(self.sg)
        self.shal = SldRow("Alone luci", 0, 100, 25); gl.addWidget(self.shal)
        self.ssoft = SldRow("Morbidezza", 0, 100, 25); gl.addWidget(self.ssoft)
        self.sv = SldRow("Vignettatura", 0, 80, 15); gl.addWidget(self.sv)
        b = QPushButton("🎲 Nuova grana"); b.setStyleSheet(BF); b.clicked.connect(self.new_variant); gl.addWidget(b)
        cl.addWidget(g)

        g = QGroupBox("🔲 Formato e inquadratura"); gl = QVBoxLayout(g)
        self.casp = QComboBox(); self.casp.addItems(list(ASPECTS)); gl.addWidget(self.casp)
        self.cori = QCheckBox("Segui orientamento della foto"); self.cori.setChecked(True); gl.addWidget(self.cori)
        self.sz = SldRow("Zoom", 100, int(MAX_ZOOM * 100), 100); gl.addWidget(self.sz)
        rw = QHBoxLayout()
        for txt, fn in (("⟲", lambda: self.rotate(-90)), ("⟳", lambda: self.rotate(90)),
                        ("Reimposta", self.reset_view)):
            b = QPushButton(txt); b.setStyleSheet(BF); b.clicked.connect(fn); rw.addWidget(b)
        gl.addLayout(rw); cl.addWidget(g)

        g = QGroupBox("💾 Output"); gl = QVBoxLayout(g)
        self.sm = SldRow("Lato lungo", 1000, 8000, 3000, "px"); gl.addWidget(self.sm)
        cl.addWidget(g)

        self.pb = QProgressBar(); self.pb.setRange(0, 100); self.pb.setTextVisible(False); cl.addWidget(self.pb)
        cl.addStretch()
        sc = QScrollArea(); sc.setWidget(inner); sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setMinimumWidth(318); sc.setMaximumWidth(360); sp.addWidget(sc)

        lb = QWidget(); ll = QVBoxLayout(lb); ll.setContentsMargins(2, 0, 2, 0)
        h = QLabel("ORIGINALE · inquadratura"); h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.setStyleSheet("color:#666;font-size:11px;font-weight:bold;letter-spacing:2px"); ll.addWidget(h)
        self.ov = OrigView(); self.ov.dropped.connect(self.add_files); ll.addWidget(self.ov, 1)
        sp.addWidget(lb)

        rb = QWidget(); rl = QVBoxLayout(rb); rl.setContentsMargins(2, 0, 2, 0)
        self.hres = QLabel("LOMOCHROME"); self.hres.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hres.setStyleSheet("color:#a56fd6;font-size:11px;font-weight:bold;letter-spacing:2px"); rl.addWidget(self.hres)
        self.fv = FilmView()
        self.fv.dropped.connect(self.add_files); self.fv.panned.connect(self._on_pan)
        self.fv.zoomed.connect(self._on_zoom); self.fv.reset.connect(self.reset_view)
        self.fv.interact.connect(self._on_interact)
        rl.addWidget(self.fv, 1); sp.addWidget(rb)
        sp.setStretchFactor(1, 2); sp.setStretchFactor(2, 3); sp.setSizes([330, 420, 640])
        root.addWidget(sp, 1)

        for w in (self.sev, self.si, self.sg, self.shal, self.ssoft, self.sv):
            w.changed.connect(lambda _=None: self._request())
        self.cfilm.currentTextChanged.connect(self._on_film)
        self.ciso.currentTextChanged.connect(lambda _: self._request())
        self.casp.currentTextChanged.connect(lambda _: self._request())
        self.cori.toggled.connect(lambda _: self._request())
        self.sz.changed.connect(lambda v: self._item() and self._on_zoom(0.5, 0.5, (v / 100) / self._item()["view"].zoom))
        self._on_film(self.cfilm.currentText())
        self._busy(False)

    # ── impostazioni ──
    def _settings(self):
        return dict(film=self.cfilm.currentText(), ev=self.sev.value() / 10, iso=int(self.ciso.currentText()),
                    strength=self.si.value() / 100, grain=self.sg.value(),
                    halation=self.shal.value() / 100, softness=self.ssoft.value() / 100,
                    vignette=self.sv.value() / 100, aspect=self.casp.currentText(),
                    auto_orient=self.cori.isChecked())

    def _on_film(self, name):
        f = FILMS[name]
        self.ldesc.setText(f.desc)
        self.sg.set_silent(int(f.grain)); self.shal.set_silent(int(f.halation * 100))
        self.hres.setText(name.upper())
        self.ciso.setEnabled(f.kind == "matrix")
        self._request()

    # ── file ──
    def open_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Apri immagini", "",
            "Immagini (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;Tutti (*)")
        if ps: self.add_files(ps)

    def add_files(self, paths):
        known = {it["path"] for it in self.items}
        for p in paths:
            if p in known: continue
            self.items.append(dict(path=p, view=View(seed=zlib.crc32(Path(p).name.encode())), proxy=None, rc={}))
            self.flist.addItem(Path(p).name)
        if self.items and self.flist.currentRow() < 0:
            self.flist.setCurrentRow(0)
        self._busy(False)

    def clear_list(self):
        self.items.clear(); self.flist.clear(); self.cur = -1
        self.ov.clear(); self.fv.clear(); self.pb.setValue(0); self._busy(False)

    def _item(self):
        return self.items[self.cur] if 0 <= self.cur < len(self.items) else None

    def _src(self, it):
        if it["proxy"] is None:
            im = load_image(it["path"]); im.thumbnail((2400, 2400), Image.LANCZOS); it["proxy"] = im
        rot = it["view"].rot
        if rot not in it["rc"]:
            o = oriented(it["proxy"], rot); it["rc"][rot] = (o, pil2pm(o))
        return it["rc"][rot]

    def _on_select(self, row):
        self.cur = row
        it = self._item()
        if not it: return
        try:
            self._src(it); self.sz.set_silent(int(round(it["view"].zoom * 100))); self._request()
        except Exception as e:
            self.statusBar().showMessage(f"Errore: {e}")

    # ── inquadratura ──
    def _geom(self):
        it = self._item(); src, _ = self._src(it)
        return it["view"], src.size, frame_aspect(self._settings(), src.size)

    def _on_pan(self, fx, fy):
        if not self._item(): return
        v, (W, H), asp = self._geom(); bw, bh = base_window(W, H, asp)
        v.cx -= fx * bw / v.zoom / W; v.cy -= fy * bh / v.zoom / H
        clamp_view(W, H, asp, v); self._request(fast=True)

    def _on_zoom(self, u, vv, factor):
        if not self._item(): return
        v, (W, H), asp = self._geom(); bw, bh = base_window(W, H, asp)
        px = v.cx * W + (u - 0.5) * bw / v.zoom; py = v.cy * H + (vv - 0.5) * bh / v.zoom
        v.zoom = min(max(v.zoom * factor, 1.0), MAX_ZOOM)
        v.cx = (px - (u - 0.5) * bw / v.zoom) / W; v.cy = (py - (vv - 0.5) * bh / v.zoom) / H
        clamp_view(W, H, asp, v)
        self.sz.set_silent(int(round(v.zoom * 100))); self._request(fast=True)

    def reset_view(self):
        it = self._item()
        if not it: return
        v = it["view"]; v.zoom, v.cx, v.cy = 1.0, 0.5, 0.5
        self.sz.set_silent(100); self._request()

    def rotate(self, deg):
        it = self._item()
        if not it: return
        v = it["view"]; v.rot = (v.rot + deg) % 360; v.cx = v.cy = 0.5; self._request()

    def new_variant(self):
        it = self._item()
        if it: it["view"].seed = random.randrange(1 << 31); self._request()

    def _on_interact(self, on):
        self._dragging = on
        if not on: self._request()

    # ── anteprima ──
    def _request(self, fast=False):
        if fast:
            self._want_fast = True; self.rt.start(10); self.ft.start(320)
        else:
            self._want_fast = False; self.rt.start(40)

    def _settle(self):
        if self._dragging: self.ft.start(320)
        else: self._request()

    def _render_preview(self):
        it = self._item()
        if not it: return
        try:
            src, spm = self._src(it); s = self._settings(); fast = self._want_fast
            img = render(src, s, it["view"], PREVIEW_FAST if fast else PREVIEW_FULL, fast)
            self.fv.set_result(pil2pm(img))
            x0, y0, x1, y1 = crop_box(src.size, frame_aspect(s, src.size), it["view"])
            W, H = src.size
            self.ov.set_data(spm, (x0 / W, y0 / H, x1 / W, y1 / H))
            if not fast:
                v = it["view"]
                self.statusBar().showMessage(f"{Path(it['path']).name}  ·  zoom {v.zoom:.2f}×  ·  rot {v.rot}°")
        except Exception as e:
            self.statusBar().showMessage(f"⚠️ {e}")

    # ── export ──
    def _busy(self, on):
        self.bsave.setEnabled(bool(self.items) and not on)
        self.ball.setEnabled(len(self.items) > 1 and not on)

    def _start(self, jobs):
        self._busy(True); self.pb.setValue(0)
        self.wk = ExportWorker(jobs, self._settings(), self.sm.value())
        self.wk.progress.connect(lambda m, p: (self.pb.setValue(p), self.statusBar().showMessage(m)))
        self.wk.done.connect(self._done); self.wk.start()

    def _suffix(self):
        return "_" + self.cfilm.currentText().replace("LomoChrome ", "").replace("'", "").replace(" ", "-").lower()

    def save_current(self):
        it = self._item()
        if not it or (self.wk and self.wk.isRunning()): return
        df = Path(it["path"]).stem + self._suffix() + ".jpg"
        p, _ = QFileDialog.getSaveFileName(self, "Salva", df, "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tiff)")
        if p: self._start([(it["path"], replace(it["view"]), p)])

    def export_all(self):
        if not self.items or (self.wk and self.wk.isRunning()): return
        d = QFileDialog.getExistingDirectory(self, "Cartella di output")
        if not d: return
        sfx = self._suffix()
        self._start([(it["path"], replace(it["view"]), os.path.join(d, Path(it["path"]).stem + sfx + ".jpg"))
                     for it in self.items])

    def _done(self, n, errs):
        self._busy(False); self.pb.setValue(100)
        if errs:
            QMessageBox.warning(self, "Esportazione", f"{n - len(errs)}/{n} salvate.\n\n" + "\n".join(errs))
        else:
            self.statusBar().showMessage(f"✅ {n} immagini salvate")
            if n > 1: QMessageBox.information(self, "Esportazione", f"{n} immagini salvate.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.ColorRole.Window, "#1c1b20"), (QPalette.ColorRole.Base, "#2a2830"),
                      (QPalette.ColorRole.WindowText, "#dddddd"), (QPalette.ColorRole.Text, "#dddddd"),
                      (QPalette.ColorRole.Button, "#2a2830"), (QPalette.ColorRole.ButtonText, "#dddddd"),
                      (QPalette.ColorRole.Highlight, "#7a45a8"), (QPalette.ColorRole.HighlightedText, "#ffffff")):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
