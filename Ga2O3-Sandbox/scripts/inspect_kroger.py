import scipy.io as sio
import numpy as np

for fn in ['092724', '072924']:
    p = f'external/KROGER/Ga2O3/Ga2O3_Varley_all_defects_new_{fn}.mat'
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    d = m['defects']
    fields = [f for f in dir(d) if not f.startswith('_')]
    print("=" * 72)
    print(f"FILE {fn} | top keys: {[k for k in m if not k.startswith('__')]}")
    print(f"defects fields ({len(fields)}):")
    for f in fields:
        v = getattr(d, f)
        try:
            arr = np.asarray(v)
            info = f"shape={arr.shape} dtype={arr.dtype}"
        except Exception as e:
            info = f"<{type(v).__name__}> {e}"
        print(f"  {f:24s} {info}")
