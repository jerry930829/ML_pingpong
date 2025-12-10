"""
RandomForest based MLPlay for 1P. Loads model from ml/models/rf_1p.joblib
"""
import os
import joblib
import pandas as pd
from ml.landing import get_predicted_landing

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_1p.joblib')

ACTION_ORDER = ['NONE', 'MOVE_LEFT', 'MOVE_RIGHT', 'SERVE_TO_LEFT', 'SERVE_TO_RIGHT']

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model not found: {MODEL_PATH}')
        data = joblib.load(MODEL_PATH)
        self.clf = data['clf']
        # Ensure parallelism at inference is limited to avoid process overhead
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
        # maintain history buffer for time-series features
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
        self_px, p1y = scene_info['platform_1P']
        opp_px, p2y = scene_info['platform_2P']
        if self.side != '1P':
            # if class used for other side accidentally, swap
            self_px, opp_px = opp_px, self_px
        blocker_x, _ = scene_info.get('blocker', (0,0))

        prev1_ball_vx = prev1_ball_vy = prev1_self_px = 0.0
        prev2_ball_vx = prev2_ball_vy = prev2_self_px = 0.0
        if len(self.history) >= 1:
            s1 = self.history[-1]
            pvx, pvy = s1['ball_speed']
            p1x, _ = s1['platform_1P']
            p2x, _ = s1['platform_2P']
            prev1_ball_vx = pvx
            prev1_ball_vy = pvy
            prev1_self_px = p1x if self.side == '1P' else p2x
        if len(self.history) >= 2:
            s2 = self.history[-2]
            pvx2, pvy2 = s2['ball_speed']
            p1x2, _ = s2['platform_1P']
            p2x2, _ = s2['platform_2P']
            prev2_ball_vx = pvx2
            prev2_ball_vy = pvy2
            prev2_self_px = p1x2 if self.side == '1P' else p2x2

        # estimate platform horizontal speeds from history (current - previous)
        p1_vx = 0
        p2_vx = 0
        if len(self.history) >= 2:
            cur = self.history[-1]
            prev = self.history[-2]
            p1_vx = cur['platform_1P'][0] - prev['platform_1P'][0]
            p2_vx = cur['platform_2P'][0] - prev['platform_2P'][0]
        try:
            res = get_predicted_landing(scene_info, p1_vx=p1_vx, p2_vx=p2_vx, return_steps=False)
            # Simplified: only request scalar pred_landing_x, no return_steps to save computation
            if isinstance(res, tuple):
                pred_landing_x = res[0]
            else:
                pred_landing_x = res
            time_to_land = None  # skip time_to_land for speed
        except Exception:
            pred_landing_x = ball_x
            time_to_land = None

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
        action = self.int_to_action[int(pred)]

        # Post-processing: conservative smoothing and landing-based override
        # Track last move to avoid rapid flips
        frame = scene_info.get('frame', 0)
        if not hasattr(self, '_last_move_frame'):
            self._last_move_frame = -999
            self._last_move_dir = None

        # If model suggests a move action, refine with landing prediction
        if action in ('MOVE_LEFT', 'MOVE_RIGHT', 'NONE'):
            # choose desired by landing dx
            landing_dx = pred_landing_x - self_px
            # Increased threshold from 8 to 15 to account for slice ball edge cases
            # and provide better margin for prediction error
            if abs(landing_dx) <= 3:
                desired = 'NONE'
            else:
                desired = 'MOVE_LEFT' if self_px > pred_landing_x else 'MOVE_RIGHT'

                # smoothing: avoid flipping direction too frequently
                if desired in ('MOVE_LEFT', 'MOVE_RIGHT') and self._last_move_dir and desired != self._last_move_dir:
                    if frame - self._last_move_frame < 4:
                        # keep previous move to avoid oscillation
                        final_action = self._last_move_dir
                    else:
                        final_action = desired
                else:
                    final_action = desired
                    
            if desired == 'NONE':
                final_action = 'NONE'

            # update last move record
            if final_action in ('MOVE_LEFT', 'MOVE_RIGHT'):
                self._last_move_dir = final_action
                self._last_move_frame = frame
        else:
            final_action = action

        # Safety override: if ball moving down toward 1P and predicted landing is left of platform,
        # but final action is NONE or MOVE_RIGHT (i.e., not moving left), force MOVE_LEFT to avoid being stuck on right edge.
        try:
            bvx, bvy = scene_info['ball_speed']
            if bvy > 0 and pred_landing_x < self_px - 8 and final_action != 'MOVE_LEFT':
                final_action = 'MOVE_LEFT'
        except Exception:
            pass

        return final_action

    def reset(self):
        pass
