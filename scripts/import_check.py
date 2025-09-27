import importlib, sys
errors=[]
for m in ["models.data_models","models.analytics","models.monitor"]:
    try:
        importlib.import_module(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"ERR: {m} -> {e}")
        errors.append((m,str(e)))
if errors:
    sys.exit(1)
else:
    print('All imports OK')
