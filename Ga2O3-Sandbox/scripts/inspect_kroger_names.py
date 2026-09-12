import scipy.io as sio
import numpy as np

p = 'external/KROGER/Ga2O3/Ga2O3_Varley_all_defects_new_092724.mat'
m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
d = m['defects']


def decode_string_field(v):
    """MATLAB string arrays land in scipy as a struct with fields s0,s1,s2,arr.
    Try several decodings and return whatever looks like readable strings."""
    out = {}
    try:
        names = v.dtype.names
        out['subfields'] = names
        for nm in names:
            sub = v[nm] if v.dtype.names else None
    except Exception as e:
        out['err'] = str(e)
    return out


for fld in ['elementnames', 'mu_names', 'mu_names_with_units', 'defect_names', 'chargestate_names']:
    v = getattr(d, fld)
    print("=" * 72)
    print("FIELD:", fld, "| type:", type(v).__name__, "| dtype:", getattr(v, 'dtype', None))
    arr = np.asarray(v)
    print("  shape:", arr.shape)
    # It's a 0-d/1-d structured array; access element
    rec = arr.reshape(-1)[0]
    print("  record fields:", rec.dtype.names if rec.dtype.names else None)
    for nm in (rec.dtype.names or []):
        sub = rec[nm]
        suba = np.asarray(sub)
        print(f"    {nm}: shape={suba.shape} dtype={suba.dtype}")
        flat = suba.reshape(-1)
        # try to show a preview
        if suba.dtype.kind in ('U', 'S', 'O'):
            preview = [str(x)[:20] for x in flat[:30]]
            print("       preview:", preview)
        elif suba.dtype.kind in ('u', 'i'):
            print("       uint preview:", flat[:60].tolist())
            # try decoding as chars
            try:
                chars = ''.join(chr(c) for c in flat if 0 < c < 0x110000)
                print("       as-chars:", repr(chars[:200]))
            except Exception as e:
                print("       chardecode err:", e)
