import joblib
import pandas as pd
import time
import numpy as np

model_data = joblib.load('ml/models/rf_1p.joblib')
clf = model_data['clf']
features = model_data['features']

print('New model info:')
print(f'  n_estimators: {clf.n_estimators}')
print(f'  max_depth: {clf.max_depth}')
print(f'  n_jobs: {clf.n_jobs}')

X_dummy = pd.DataFrame(np.random.randn(1, len(features)), columns=features)

start = time.time()
for _ in range(1000):
    _ = clf.predict(X_dummy)
elapsed = time.time() - start
ms_per_predict = elapsed / 1000 * 1000
print(f'1000x single predict: {elapsed:.3f}s ({ms_per_predict:.3f}ms per predict)')
if ms_per_predict < 16.67:
    print(f'✓ For 60fps need: 16.67ms per frame; predict cost: {ms_per_predict:.3f}ms OK')
else:
    print(f'✗ Still too slow: {ms_per_predict:.3f}ms > 16.67ms')
