

from functools import lru_cache
import pygame
from mlgame.game import physics
from src.game_object import PLATFORM_W, PLATFORM_H


# ------------------------------------------------------------
#                 Internal Simulation (Cached)
# ------------------------------------------------------------
@lru_cache(maxsize=32768)
def _cached_simulate(qkey):
    """
    qkey = (bx, by, bvx, bvy, p1x, p1y, p2x, p2y, blocker_x, blocker_y)
   
    """
    (bx, by, bvx, bvy,
     p1x, p1y, p2x, p2y,
     bxk, byk) = qkey

    # --- reconstruct rects ---
    ball_rect = pygame.Rect(bx, by, 10, 10)
    ball_speed = [bvx, bvy]

    box_rect = pygame.Rect(0, 0, 200, 500)

    p1_rect = pygame.Rect(p1x, p1y, PLATFORM_W, PLATFORM_H)
    p2_rect = pygame.Rect(p2x, p2y, PLATFORM_W, PLATFORM_H)

    blocker_rect = None
    if not (bxk == 0 and byk == 0):
        blocker_rect = pygame.Rect(bxk, byk, 30, 20)

    # --- moving object replica ---
    class _M:
        pass

    moving = _M()
    moving.rect = ball_rect.copy()
    moving.last_pos = ball_rect.copy()

    MAX_STEPS = 300  

    for _ in range(MAX_STEPS):

        # step
        moving.last_pos = moving.rect.copy()
        moving.rect = moving.rect.copy()
        moving.rect.move_ip(ball_speed)

        # bounce in box
        new_rect, new_speed = physics.bounce_in_box(moving.rect, ball_speed, box_rect)
        moving.rect = new_rect
        ball_speed = new_speed

        # platform & blocker collision
        hit_obj = None
        if moving.rect.colliderect(p1_rect):
            hit_obj = ("p1", p1_rect)
        elif moving.rect.colliderect(p2_rect):
            hit_obj = ("p2", p2_rect)
        elif blocker_rect and moving.rect.colliderect(blocker_rect):
            hit_obj = ("blocker", blocker_rect)


        if hit_obj is not None:
            tag, hit_rect = hit_obj

            # decide hit-speed
            hit_speed = [0, 0]
            if tag == "p1":
                pass
            elif tag == "p2":
                pass
            # blocker speed = [0,0]

            new_r, new_s = physics.bounce_off(moving.rect, ball_speed, hit_rect, hit_speed)
            moving.rect = new_r
            ball_speed = new_s

            if tag in ("p1", "p2"):
                return float(moving.rect.centerx)

        # --- Anti-tunneling ---
        # If last frame is above platform and this frame is below platform, but not colliderect → manually land
        # (This prevents extremely fast ball skipping through thin platform)
        if (moving.last_pos.bottom < p1_rect.top and moving.rect.bottom >= p1_rect.top) and \
                not moving.rect.colliderect(p1_rect):
            return float(moving.rect.centerx)

        if (moving.last_pos.top > p2_rect.bottom and moving.rect.top <= p2_rect.bottom) and \
                not moving.rect.colliderect(p2_rect):
            return float(moving.rect.centerx)

    # fallback
    return float(moving.rect.centerx)


# ------------------------------------------------------------
#               Public API (Compatible with old code)
# ------------------------------------------------------------

_LAST_FRAME = None
_LAST_PRED = None


def simulate_landing_x(scene, max_steps=200, p1_vx=0, p2_vx=0, return_steps=False):

    pred = get_predicted_landing(scene)

    if return_steps:
        return pred, None
    return pred


def get_predicted_landing(scene, max_steps=200, p1_vx=0, p2_vx=0, return_steps=False):
    """
    外部 1P / 2P agent 使用的 API
    """

    global _LAST_FRAME, _LAST_PRED

    frame = scene.get("frame")

    # --- frame cache (for 1P + 2P same frame) ---
    if frame is not None and frame == _LAST_FRAME:
        return _LAST_PRED if not return_steps else (_LAST_PRED, None)

    # --- extract scene for simulation ---
    bx, by = scene["ball"]
    bvx, bvy = scene["ball_speed"]

    p1x, p1y = scene["platform_1P"]
    p2x, p2y = scene["platform_2P"]

    blocker = scene.get("blocker", (0, 0))
    bxk, byk = blocker if blocker else (0, 0)

    # quantize into ints for caching
    q = lambda v: int(round(v))

    qkey = (
        q(bx), q(by), q(bvx), q(bvy),
        q(p1x), q(p1y), q(p2x), q(p2y),
        q(bxk), q(byk)
    )

    pred = _cached_simulate(qkey)

    if frame is not None:
        _LAST_FRAME = frame
        _LAST_PRED = pred

    return pred if not return_steps else (pred, None)
