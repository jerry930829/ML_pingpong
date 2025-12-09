from mlgame.game import physics
import pygame
from src.game_object import PLATFORM_W, PLATFORM_H


def simulate_landing_x(scene, max_steps=100, p1_vx=0, p2_vx=0, return_steps=False):
    """Simulate ball forward using game physics (bounce-aware) and return landing x.

    This uses the same physics helpers as the game, but treats platforms/blocker
    as static for prediction. Returns float x in scene coordinates.
    Reduced default max_steps to 100 for speed (sufficient for most cases).
    """
    bx, by = scene['ball']
    bvx, bvy = scene['ball_speed']
    p1x_, p1y_ = scene['platform_1P']
    p2x_, p2y_ = scene['platform_2P']

    # ball and object sizes are from README
    ball_rect = pygame.Rect(int(bx), int(by), 10, 10)
    ball_speed = [int(bvx), int(bvy)]
    box_rect = pygame.Rect(0, 0, 200, 500)

    p1_rect = pygame.Rect(int(p1x_), int(p1y_), PLATFORM_W, PLATFORM_H)
    p2_rect = pygame.Rect(int(p2x_), int(p2y_), PLATFORM_W, PLATFORM_H)
    blocker_pos = scene.get('blocker', (0, 0))
    blocker_rect = None
    if blocker_pos and blocker_pos != (0, 0):
        bxk, byk = blocker_pos
        blocker_rect = pygame.Rect(int(bxk), int(byk), 30, 20)

    class _Moving:
        pass

    moving = _Moving()
    moving.rect = ball_rect.copy()
    moving.last_pos = ball_rect.copy()

    for step in range(max_steps):
        # step
        moving.last_pos = moving.rect.copy()
        moving.rect = moving.rect.copy()
        moving.rect.move_ip(ball_speed)

        # bounce inside box
        new_rect, new_speed = physics.bounce_in_box(moving.rect, ball_speed, box_rect)
        moving.rect = new_rect
        ball_speed = new_speed

        # check collision with platforms/blocker (use fast rect.colliderect)
        hit = None
        if moving.rect.colliderect(p1_rect):
            hit = p1_rect
        elif moving.rect.colliderect(p2_rect):
            hit = p2_rect
        elif blocker_rect and moving.rect.colliderect(blocker_rect):
            hit = blocker_rect

        if hit is not None:
            # compute bounce off result (use platform horizontal speed to model slice)
            hit_speed = [0, 0]
            if hit is p1_rect:
                hit_speed = [int(p1_vx), 0]
            elif hit is p2_rect:
                hit_speed = [int(p2_vx), 0]
            # blocker treated as static
            new_r, new_s = physics.bounce_off(moving.rect, ball_speed, hit, hit_speed)
            moving.rect = new_r
            ball_speed = new_s
            # if hit is a platform, consider this the landing
            if hit is p1_rect or hit is p2_rect:
                return (float(moving.rect.centerx), step+1) if return_steps else float(moving.rect.centerx)

        # termination: if ball crosses the platform plane without explicit collision
        if moving.last_pos.bottom < p1_rect.top <= moving.rect.bottom:
            return (float(moving.rect.centerx), step+1) if return_steps else float(moving.rect.centerx)
        if moving.last_pos.top > p2_rect.bottom >= moving.rect.top:
            return (float(moving.rect.centerx), step+1) if return_steps else float(moving.rect.centerx)

    # fallback: return current x (and steps if requested)
    return (float(moving.rect.centerx), max_steps) if return_steps else float(moving.rect.centerx)


# Simple frame-keyed cache to avoid re-simulating multiple times per game frame.
# The game provides `scene['frame']` so we use that as cache key (fast and safe
# because both agents/collector see the same scene for a frame).
_LAST_FRAME = None
_LAST_PRED = None

def get_predicted_landing(scene, max_steps=200, p1_vx=0, p2_vx=0, return_steps=False):
    global _LAST_FRAME, _LAST_PRED
    frame = scene.get('frame')
    # Use frame-level caching when available. Cache stores tuple (pred, steps) when return_steps True,
    # or scalar pred when return_steps False. We key by frame only.
    if frame is not None and frame == _LAST_FRAME:
        cached = _LAST_PRED
        if return_steps:
            # if cached is scalar, compute steps by re-running (fallback)
            if isinstance(cached, tuple):
                return cached
            else:
                # rerun to get steps
                res = simulate_landing_x(scene, max_steps=max_steps, p1_vx=p1_vx, p2_vx=p2_vx, return_steps=True)
                if frame is not None:
                    _LAST_PRED = res
                return res
        else:
            # want scalar
            if isinstance(cached, tuple):
                return cached[0]
            else:
                return cached

    # not cached or different frame
    if return_steps:
        res = simulate_landing_x(scene, max_steps=max_steps, p1_vx=p1_vx, p2_vx=p2_vx, return_steps=True)
        if frame is not None:
            _LAST_FRAME = frame
            _LAST_PRED = res
        return res
    else:
        pred = simulate_landing_x(scene, max_steps=max_steps, p1_vx=p1_vx, p2_vx=p2_vx)
        if frame is not None:
            _LAST_FRAME = frame
            _LAST_PRED = pred
        return pred
