"""py2app build script for VoicePad.

Usage (from a Python 3.11+ venv with requirements.txt + py2app installed):
    python setup.py py2app

Produces dist/VoicePad.app — a self-contained background-agent app with
its own bundled Python interpreter. Solves the libggml-metal abort dialog
by isolating from the system's homebrew Python (where llama-cpp-python
ships libggml dylibs that abort during clean shutdown).
"""
import sys
from setuptools import setup

# py2app's modulegraph blows the default 1000-deep recursion limit on
# our import graph (mlx → numpy → ...). Bumping it lets the build complete.
sys.setrecursionlimit(10000)

APP = ['voicepad.py']

OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'VoicePad',
        'CFBundleDisplayName': 'VoicePad',
        'CFBundleIdentifier': 'com.voicepad.app',
        'CFBundleVersion': '1.0',
        'CFBundleShortVersionString': '1.0',
        # LSUIElement: background agent — no Dock icon, no menu bar entry,
        # no Cmd-Tab listing. Just our floating panel.
        'LSUIElement': True,
        'NSMicrophoneUsageDescription':
            'VoicePad needs microphone access to transcribe your voice.',
        'NSAppleEventsUsageDescription':
            'VoicePad uses AppleScript to detect the focused app and paste transcribed text.',
    },
    # Packages py2app's modulegraph may miss (lazy imports, C extensions).
    # Note: 'mlx' is intentionally NOT listed here — it's a namespace package
    # (no __init__.py; the actual code is contributed by the mlx-metal
    # distribution), and py2app's legacy imp.find_module() chokes on those.
    # We copy the mlx tree into the bundle manually after build (see end of
    # this file).
    'packages': [
        'mlx_whisper',
        'sounddevice',
        'pyperclip', 'pynput', 'requests',
        'numpy', 'objc',
    ],
    'includes': [
        'AppKit', 'Foundation', 'Quartz', 'CoreFoundation',
    ],
    # Belt-and-suspenders: never bundle anything that pulls in libggml-metal,
    # even if a transitive dep references it. These are the libraries known to
    # bring it in via the homebrew Python 3.11 site-packages on this machine.
    'excludes': [
        'llama_cpp', 'llama_cpp_python',
        'whispercpp', 'pywhispercpp', 'whisper_cpp',
        'gpt4all',
        'tkinter',  # we use AppKit; tkinter would just bloat the bundle
    ],
    'optimize': 0,
    # Don't strip dylibs — MLX's Metal kernels sometimes break under strip.
    'strip': False,
    'iconfile': None,
}

setup(
    app=APP,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)


# ── post-build: replace py2app's partial mlx with the full namespace tree ────
# py2app extracts only mlx/core.so to lib-dynload/mlx and skips the namespace
# package's Python sources (_reprlib_fix.py, nn/, optimizers/, ...). At runtime
# Python finds the partial lib-dynload mlx and fails when mlx_whisper tries to
# import mlx._reprlib_fix. Strategy: blow away py2app's partial mlx and replace
# it with the full venv mlx tree at the same location (lib-dynload IS on
# sys.path, so this avoids namespace splitting and any sys.path gymnastics).
# The unmodified core.cpython-311-darwin.so already has rpath @loader_path/lib
# pointing at mlx/lib/ which contains libmlx.dylib, libjaccl.dylib, and the
# Metal kernels — so no install_name_tool dance is needed either.
if 'py2app' in sys.argv:
    import os, shutil, zipfile, importlib.util
    spec = importlib.util.find_spec('mlx')
    if spec is not None and spec.submodule_search_locations:
        mlx_src = list(spec.submodule_search_locations)[0]
        mlx_dst = 'dist/VoicePad.app/Contents/Resources/lib/python3.11/lib-dynload/mlx'
        if os.path.exists(mlx_dst):
            shutil.rmtree(mlx_dst)
        shutil.copytree(mlx_src, mlx_dst, symlinks=True)
        print(f"[post-build] replaced lib-dynload/mlx with full tree from {mlx_src}")

    # py2app archives single-file modules (like sounddevice.py) into
    # lib/python311.zip. sounddevice locates libportaudio.dylib via
    # os.path.dirname(__file__)/_sounddevice_data/, but dlopen() can't
    # read from inside a zip. Fix: extract sounddevice + its data dir to
    # disk AND rewrite the zip without those entries so Python's import
    # machinery falls through to the on-disk copy (whose __file__ is a
    # real path that dlopen can resolve).
    zip_path = 'dist/VoicePad.app/Contents/Resources/lib/python311.zip'
    extract_root = 'dist/VoicePad.app/Contents/Resources/lib/python3.11'
    if os.path.exists(zip_path):
        # 1. Extract sounddevice.py(c) + its data dir to disk.
        with zipfile.ZipFile(zip_path, 'r') as z:
            sd_members = [n for n in z.namelist()
                          if n in ('sounddevice.py', 'sounddevice.pyc')
                          or n.startswith('_sounddevice_data/')]
            for m in sd_members:
                z.extract(m, extract_root)
        # Also copy the source data dir directly in case the in-zip
        # version is missing files.
        import sounddevice
        sd_data_src = os.path.join(os.path.dirname(sounddevice.__file__), '_sounddevice_data')
        if os.path.isdir(sd_data_src):
            sd_data_dst = os.path.join(extract_root, '_sounddevice_data')
            if os.path.exists(sd_data_dst):
                shutil.rmtree(sd_data_dst)
            shutil.copytree(sd_data_src, sd_data_dst)
        # 2. Rewrite the zip without those entries so Python falls through.
        # Also strip py2app's mlx stub (`mlx/core.pyc`), which references a
        # file named `mlx/core.so` that doesn't exist — our on-disk copy is
        # named `core.cpython-311-darwin.so` (the venv's original name).
        # Removing the stub makes Python load the real extension from disk.
        tmp_zip = zip_path + '.new'
        with zipfile.ZipFile(zip_path, 'r') as src, zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if (item.filename.startswith('sounddevice')
                        or item.filename.startswith('_sounddevice_data/')
                        or item.filename.startswith('mlx/')):
                    continue
                dst.writestr(item, src.read(item.filename))
        shutil.move(tmp_zip, zip_path)
        print(f"[post-build] purged sounddevice + mlx stubs from zip; sounddevice on disk at {extract_root}")

