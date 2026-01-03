import traceback
try:
    import torchcodec
    print('torchcodec imported OK')
except Exception:
    traceback.print_exc()