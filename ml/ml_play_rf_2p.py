"""
RandomForest based MLPlay for 2P. Loads model from ml/models/rf_2p.joblib
"""
import os
import joblib
import pandas as pd
from ml.landing import simulate_landing_x

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_2p.joblib')

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model not found: {MODEL_PATH}')
        data = joblib.load(MODEL_PATH)
        self.clf = data['clf']
        # Limit parallel jobs during inference to avoid process spawning overhead
        try:
            if hasattr(self.clf, 'n_jobs'):
                self.clf.n_jobs = 1
        except Exception:
            pass
        self.action_to_int = data['action_to_int']
        self.features = data.get('features')
        if not self.features:
            self.features = ['ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x']
        self.int_to_action = {v:k for k,v in self.action_to_int.items()}

    def update(self, scene_info, *args, **kwargs):
        if scene_info['status'] != 'GAME_ALIVE':
            return 'RESET'
        # Maintain history for time-series features
        if not hasattr(self, 'history'):
            from collections import deque
            self.history = deque(maxlen=2)
        self.history.append({
            'ball_speed': scene_info['ball_speed'],
            'platform_1P': scene_info['platform_1P'],
            'platform_2P': scene_info['platform_2P'],
        })

        ball_x, ball_y = scene_info['ball']
        ball_vx, ball_vy = scene_info['ball_speed']
        p1x, p1y = scene_info['platform_1P']
        p2x, p2y = scene_info['platform_2P']
        self_px = p2x if self.side == '2P' else p1x
        opp_px = p1x if self.side == '2P' else p2x
        blocker_x, _ = scene_info.get('blocker', (0,0))

        prev1_ball_vx = prev1_ball_vy = prev1_self_px = 0.0
        prev2_ball_vx = prev2_ball_vy = prev2_self_px = 0.0
        if len(self.history) >= 1:
            s1 = self.history[-1]
            pvx, pvy = s1['ball_speed']
            p1x_, _ = s1['platform_1P']
            p2x_, _ = s1['platform_2P']
            prev1_ball_vx = pvx
            prev1_ball_vy = pvy
            prev1_self_px = p2x_ if self.side == '2P' else p1x_
        if len(self.history) >= 2:
            s2 = self.history[-2]
            pvx2, pvy2 = s2['ball_speed']
            p1x2, _ = s2['platform_1P']
            p2x2, _ = s2['platform_2P']
            prev2_ball_vx = pvx2
            prev2_ball_vy = pvy2
            prev2_self_px = p2x2 if self.side == '2P' else p1x2

        # estimate platform horizontal speeds from history (current - previous)
        p1_vx = 0
        p2_vx = 0
        if len(self.history) >= 2:
            cur = self.history[-1]
            prev = self.history[-2]
            p1_vx = cur['platform_1P'][0] - prev['platform_1P'][0]
            p2_vx = cur['platform_2P'][0] - prev['platform_2P'][0]
        try:
            pred_landing_x = simulate_landing_x(scene_info, p1_vx=p1_vx, p2_vx=p2_vx)
        except Exception:
            pred_landing_x = ball_x

        vals = {
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_vx': ball_vx,
            'ball_vy': ball_vy,
            'self_px': self_px,
            'opp_px': opp_px,
            'ball_served': int(scene_info['ball_served']),
            'serving_is_self': int(scene_info['serving_side']==self.side),
            'blocker_x': blocker_x,
            'pred_landing_x': pred_landing_x,
            'prev1_ball_vx': prev1_ball_vx,
            'prev1_ball_vy': prev1_ball_vy,
            'prev1_self_px': prev1_self_px,
            'prev2_ball_vx': prev2_ball_vx,
            'prev2_ball_vy': prev2_ball_vy,
            'prev2_self_px': prev2_self_px,
        }

        df = pd.DataFrame([vals], columns=self.features)
        pred = self.clf.predict(df)[0]
        return self.int_to_action[int(pred)]

    def reset(self):
        pass
