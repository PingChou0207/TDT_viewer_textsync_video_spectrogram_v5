# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

project_root = Path(SPECPATH)
entry_script = project_root / "src" / "tdt_viewer_textsync_lfp_spectrogram_v4.py"

a = Analysis(
    [str(entry_script)],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[],
    hiddenimports=["tdt.TDTbin2py", "tdt.TDTfilter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6",
        "PyQt5",
        "PySide2",
        "matplotlib",
        "pandas",
        "IPython",
        "jupyter",
        "pytest",
        "torch",
        "OpenGL",
        "numba",
        "llvmlite",
        "sphinx",
        "docutils",
        "lxml",
        "PIL",
        "jinja2",
        "babel",
        "pytz",
        "cryptography",
        "bcrypt",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TDT Viewer TextSync LFP Spectrogram v4",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TDT Viewer TextSync LFP Spectrogram v4",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="TDT Viewer TextSync LFP Spectrogram v4.app",
        icon=None,
        bundle_identifier="com.pingchou.tdt-viewer-textsync-lfp-spectrogram-v4",
        info_plist={
            "CFBundleDisplayName": "TDT Viewer TextSync LFP Spectrogram v4",
            "CFBundleShortVersionString": "4.0.0",
            "CFBundleVersion": "4.0.0",
            "NSHighResolutionCapable": True,
        },
    )
