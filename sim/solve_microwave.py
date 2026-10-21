#!/usr/bin/env python
"""Scripted solution of MicrowavePot-v0: every arm motion is solved by inverse kinematics
along a densely interpolated Cartesian path (SAPIEN's pinocchio IK, each point seeded
with the last) and executed in the physics simulation with joint-position control, so a
run either works physically or shows where it does not. (ManiSkill's mplib planner
segfaults here: mplib 0.1.1 is built for numpy 1.x, the env has numpy 2.4.)

Task structure and targets come from the human takes (recon/DEMOS.md): open the door,
pick the pot up, place it on the front half of the turntable, shut the door. Grasp
geometry comes from the measured assets.

  1. door: grasp the vertical bar handle from the right, along the door face, one finger
     between bar and door, and carry the grasp round the hinge, 1 deg at a time
  2. pot: grasp its short side from the front at body height (under the rim), lift
  3. place: through the opening, along +y_mw, to the place target; open; back out
  4. door: grasp the handle again and carry it round back to 0

Obstacles are avoided by via points (a pose above the counter between phases), not by
a planner. Video, trajectory and a json of phase outcomes go to --out.

  python sim/solve_microwave.py --out <dir> [--open-deg 90] [--seed 0]
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.quaternions import qmult, qinverse, quat2axangle, axangle2quat, quat2mat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sim.microwave_env as M  # noqa: E402
import sim.human_paths as HP  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

HANDLE_GRASP_Z = 0.17        # on the bar, between its centre (0.143) and upper post (0.223)
POT_GRASP_Z = 0.050          # under the rim (top at 0.0635); the flange clears the counter
PLACE_MW = (0.18, 0.09)      # front half of the turntable, from the takes
READY_MW = (0.30, -0.12, 0.40)                         # above the front edge, clear of the handle
NOMINAL_Q = [0.0, 0.2, 0.0, -2.0, 0.0, 2.2, np.pi / 4]  # a comfortable arm for the ready IK
TUCK_MW = (0.15, -0.38, 0.45)  # high, in front of the robot, hand down: crossings start here
DOWN_Q = np.array([0.0, 1.0, 0.0, 0.0])   # wxyz: tcp z pointing down (world)
OPEN, CLOSED = 1.0, -1.0
HALF_OPEN = 0.0              # fingers 15 mm each: the bar sits 21.5 mm in front of the door


def mw_dir(v):
    return M.R_W_MW @ np.asarray(v, float)


def door_rot(q):
    """Door frame -> mw frame at door angle q (hinge axis -z through the origin)."""
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def slerp(q0, q1, t):
    d = qmult(q1, qinverse(q0))
    if d[0] < 0:
        d = -d
    ax, ang = quat2axangle(d)
    return qmult(axangle2quat(ax, ang * t), q0)


class Arm:
    """IK along Cartesian paths, executed with pd_joint_pos."""

    def __init__(self, env, obs=None, noise=0.0, rng_seed=0, record_states=False, speed=1.0, j5_max=None):
        """noise > 0: DART-style data. The executed joint targets get a smooth random
        offset (std `noise` rad), while the recorded action stays the planned, clean
        target -- so the data shows how to get back onto the plan from where the noise
        put the arm. Halved while the gripper is closed on something."""
        self.env, self.u = env, env.unwrapped
        # speed > 1: every path is followed through every speed-th of its IK points and joint-space
        # moves take speed-times larger steps; the gripper's closing / opening time is unchanged
        self.speed = speed
        self.jumps = []                  # every joint-space transfer: step index, |dq| per joint (deg)
        self.closest_first = j5_max is not None   # --smooth: among complete plans, the closest to the arm
        self.noise, self.rng = noise, np.random.default_rng(rng_seed)
        self.offset = np.zeros(7)
        self.last_obs = None if obs is None else np.asarray(obs.cpu() if hasattr(obs, "cpu") else obs).reshape(-1)
        self.rec_obs, self.rec_act = [], []
        # record_states: also the simulator state behind every recorded observation, so the
        # demonstration can be re-rendered from any camera later (sim/render_views.py)
        self.record_states, self.rec_state = record_states, []
        self.robot = self.u.agent.robot._objs[0]
        self.pm = self.robot.create_pinocchio_model()
        self.ee = [i for i, l in enumerate(self.robot.links) if l.name.endswith("panda_hand_tcp")][0]
        self.q_cmd = self.robot.get_qpos()[:7].copy()
        self.grip = OPEN
        self.lim = self.robot.get_qlimits()[:7].copy()
        if j5_max is not None:           # keep the wrist on one branch: no j5 flips between phases
            self.lim[4] = [-j5_max, j5_max]
        self.fingers = self.robot.get_active_joints()[7:]
        self.default_drive = [(j.stiffness, j.damping, j.force_limit) for j in self.fingers]

    def firm_grip(self, on):
        """ManiSkill's finger drive (stiffness 1e3) squeezes the pot with ~37 N, and a held
        pot turned about the finger axis during joint-space moves; 1e4 / 200 N gives ~140 N,
        the real Panda's peak. On the thin handle bar that much makes the contact unstable
        (the bar slips out), so it is only for the pot."""
        for j, (k, d, f) in zip(self.fingers, self.default_drive):
            if on:
                j.set_drive_properties(stiffness=1e4, damping=d, force_limit=200)
            else:
                j.set_drive_properties(stiffness=k, damping=d, force_limit=f)

    def ik(self, P, seed, restarts=12, prefer_q=None):
        """Joint angles for tcp pose P within the joint limits, closest to the seed.
        pinocchio's IK ignores limits; when its answer is outside them, a bounded
        least-squares solve runs from the seed and from random starts."""
        q, ok, _ = self.pm.compute_inverse_kinematics(
            self.ee, P, initial_qpos=seed, active_qmask=[1] * 7 + [0, 0], max_iterations=300)
        if ok and np.all(q[:7] >= self.lim[:, 0]) and np.all(q[:7] <= self.lim[:, 1]):
            return q[:7]
        from scipy.optimize import minimize
        Rt = quat2mat(P.q)
        full = seed.copy()

        def err(x):
            full[:7] = x
            self.pm.compute_forward_kinematics(full)
            F = self.pm.get_link_pose(self.ee)
            dR = Rt.T @ quat2mat(F.q)
            return np.linalg.norm(F.p - P.p), np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))

        prefer = seed[:7] if prefer_q is None else np.asarray(prefer_q)

        def cost(x):
            dp, ang = err(x)
            return dp ** 2 + 0.01 * ang ** 2

        def cost_natural(x):                          # + a pull towards a natural arm
            return cost(x) + 1e-4 * np.sum((x - prefer) ** 2)
        rng = np.random.default_rng(0)
        starts = [np.clip(seed[:7], self.lim[:, 0] + 1e-3, self.lim[:, 1] - 1e-3)]
        starts += list(rng.uniform(self.lim[:, 0], self.lim[:, 1], size=(restarts, 7)))
        best = None
        for x0 in starts:
            # FK is float32: finite differences need a step far above 1e-8
            # natural basin first, then the exact pose from there
            r = minimize(cost_natural, x0, method="L-BFGS-B", bounds=self.lim,
                         options=dict(maxiter=500, eps=1e-4))
            r = minimize(cost, r.x, method="L-BFGS-B", bounds=self.lim,
                         options=dict(maxiter=500, eps=1e-4))
            dp, ang = err(r.x)
            if dp < 0.002 and ang < np.deg2rad(2):
                d = np.abs(r.x - seed[:7]).sum()
                if best is None or d < best[0]:
                    best = (d, r.x)
                if x0 is starts[0]:
                    break                                    # the seed itself works: closest
        return None if best is None else best[1]

    def candidates(self, P, n=24, seed=0):
        """Distinct joint solutions for one pose, from random starts (the arm has one
        redundant degree of freedom, so they can differ a lot)."""
        from scipy.optimize import minimize
        Rt = quat2mat(P.q)
        full = np.r_[self.q_cmd, 0.04, 0.04]

        def err(x):
            full[:7] = x
            self.pm.compute_forward_kinematics(full)
            F = self.pm.get_link_pose(self.ee)
            dR = Rt.T @ quat2mat(F.q)
            return np.linalg.norm(F.p - P.p), np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        out = []
        for x0 in np.random.default_rng(seed).uniform(self.lim[:, 0], self.lim[:, 1], size=(n, 7)):
            r = minimize(lambda x: err(x)[0] ** 2 + 0.01 * err(x)[1] ** 2, x0, method="L-BFGS-B",
                         bounds=self.lim, options=dict(maxiter=500, eps=1e-4))
            dp, ang = err(r.x)
            if dp < 0.002 and ang < np.deg2rad(2) and all(np.abs(r.x - o).max() > 0.2 for o in out):
                out.append(r.x)
        return out

    def path(self, poses, q_start, max_jump=0.3):
        """Joint path through poses, each IK seeded with the previous; how far it got."""
        qs, qi = [], np.r_[q_start, 0.04, 0.04]
        prev = np.asarray(q_start)
        for P in poses:
            q = self.ik(P, qi)
            if q is None or np.abs(q - prev).max() > max_jump:
                return qs
            qs.append(q)
            prev, qi = q, np.r_[q, qi[7:]]
        return qs

    def plan(self, poses, n_candidates=24):
        """Joint path through a pose sequence: IK candidates at the first pose, each
        continued through the rest; the one that gets furthest (then: furthest from the
        joint limits, then: closest to the current arm) wins."""
        best = None
        for qc in self.candidates(poses[0], n=n_candidates):
            qs = [qc] + self.path(poses[1:], qc)
            # whole sequence first, then the closest to the current arm (a short, predictable
            # joint-space move), then clear of the joint limits
            key = (len(qs), self.margin(qs) > 0.05, -np.abs(qc - self.q_cmd).sum()) if self.closest_first else \
                (len(qs), -np.round(np.abs(qc - self.q_cmd).sum(), 1), self.margin(qs) > 0.05)
            if best is None or key > best[0]:
                best = (key, qs)
        return [] if best is None else best[1]

    def run(self, poses, settle=8):
        """Plan through poses, go to the first in joint space, follow the rest."""
        qs = self.plan(poses)
        if len(qs) < len(poses):
            return False
        self.move_joint(qs[0])
        return self.execute(qs[1:], settle=settle)

    def line(self, target, start=None, step_pos=0.01, step_rot=np.deg2rad(4)):
        """Poses from start (default: the tcp) to target: straight, rotation slerped."""
        s0 = self.tcp() if start is None else start
        dp = np.linalg.norm(target.p - s0.p)
        d = qmult(target.q, qinverse(s0.q))
        ang = abs(quat2axangle(d if d[0] >= 0 else -d)[1])
        n = int(max(np.ceil(dp / step_pos), np.ceil(ang / step_rot), 1))
        return [sapien.Pose(s0.p + (target.p - s0.p) * k / n, slerp(s0.q, target.q, k / n))
                for k in range(1, n + 1)]

    def go(self, poses, settle=8, high=False):
        """Follow poses from the current arm by continuation (the hand moves along them).
        If that breaks: high=True (a transfer above everything) goes to the last pose in
        joint space, by its IK solution closest to the current arm; otherwise plan with IK
        candidates and get to the first pose in joint space."""
        poses = self.line(poses[0]) + list(poses[1:])
        qs = self.path(poses, self.q_cmd)
        if len(qs) == len(poses):
            return self.execute(qs, settle=settle)
        print(f"    continuation stopped at {len(qs)}/{len(poses)}; "
              + ("joint-space transfer" if high else "replanning"), flush=True)
        if high:
            cands = self.candidates(poses[-1])
            if not cands:
                return False
            q = min(cands, key=lambda c: np.abs(c - self.q_cmd).sum())
            return self.move_joint(q, step=0.02)
        return self.run(poses, settle=settle)

    def margin(self, qs):
        q = np.array(qs)
        return float(np.min(np.minimum(q - self.lim[:, 0], self.lim[:, 1] - q)))

    def execute(self, qs, settle=8):
        if self.speed != 1.0 and len(qs) > 1:
            n = len(qs)
            keep = np.unique(np.r_[np.round(np.arange(self.speed - 1, n, self.speed)).astype(int), n - 1])
            qs = [qs[i] for i in keep if i < n]
            settle = max(3, int(round(settle / self.speed)))
        for q in qs:
            self.q_cmd = np.asarray(q)
            self.act()
        self.act(settle)
        return True

    def tcp(self):
        return self.u.agent.tcp.pose.sp

    def act(self, n=1):
        for _ in range(n):
            clean = np.r_[self.q_cmd, self.grip]
            executed = clean.copy()
            if self.noise > 0:
                scale = 0.5 if self.grip < -0.5 else 1.0
                self.offset = 0.9 * self.offset + np.sqrt(1 - 0.9 ** 2) * self.rng.normal(0, self.noise, 7)
                executed[:7] += scale * self.offset
            if self.last_obs is not None:
                self.rec_obs.append(self.last_obs)
                self.rec_act.append(clean)
                if self.record_states:
                    self.rec_state.append(self.u.get_state().cpu().numpy()[0])
            obs = self.env.step(executed)[0]
            self.last_obs = np.asarray(obs.cpu() if hasattr(obs, "cpu") else obs).reshape(-1)

    def move_joint(self, q_target, step=0.03):
        """Straight line in joint space (for big re-orientations away from obstacles)."""
        q0 = self.q_cmd.copy()
        self.jumps.append(dict(t=len(self.rec_act), dq_deg=np.degrees(np.abs(np.asarray(q_target) - q0)).round(0).tolist()))
        step *= self.speed
        n = int(max(np.ceil(np.abs(q_target - q0).max() / step), 1))
        for k in range(1, n + 1):
            self.q_cmd = q0 + (q_target - q0) * k / n
            self.act()
        self.act(max(3, int(round(8 / self.speed))))
        return True

    def gripper(self, state, n=15):
        self.grip = state
        self.act(n)


class Solver:
    """The task's phases as methods on a live environment, so they can also be started
    from a state the solver did not produce (sim/dagger.py hands over mid-episode)."""

    def __init__(self, env, arm):
        self.env, self.u, self.arm = env, env.unwrapped, arm
        self.der = json.load(open(M.MW_PARAMS))["derived"]
        hx = self.der["handle_bar_centre_closed"]
        self.handle_door = np.array([hx[0], hx[1], HANDLE_GRASP_Z])     # in the door frame (= mw at q=0)
        self.log = []
        self.q_tuck = None
        self.flip = False                # the hand turned 180 deg about its axis: the same grasp (symmetric fingers)
        self.try_flip = False            # --smooth: plan each grasp phase with both and keep the closer one
        self.handle_plan = None

    def step(self, name, ok):
        info = self.u.evaluate()
        rec = dict(phase=name, solved=bool(ok), door_deg=float(info["door_deg"][0]),
                   pot_mw=np.round(self.u.pot_in_mw().cpu().numpy()[0], 3).tolist(),
                   t=len(self.arm.rec_obs))           # recorded steps so far: the phase ends here
        self.log.append(rec)
        print(rec, flush=True)
        return ok

    def grasp_pose(self, center_mw, approach_mw, closing_mw):
        clo = -np.asarray(closing_mw, float) if self.flip else closing_mw
        return self.u.agent.build_grasp_pose(mw_dir(approach_mw), mw_dir(clo), M.mw_to_world(center_mw))

    def flips(self):
        return (False, True) if self.try_flip else (self.flip,)

    def door_dirs(self, q):
        Rq = door_rot(q)
        return Rq @ np.array([0, 1.0, 0]), Rq @ np.array([1.0, 0, 0])

    def handle_pose(self, q, back=0.0, up=0.0):
        # square to the door, fingers across the bar. (Approaching from the side along the
        # door face, one finger between bar and door, was tried: the arc then only reaches
        # 55 deg and the grip does not hold.)
        Rq = door_rot(q)
        app = Rq @ np.array([0, 1.0, 0])
        clo = Rq @ np.array([1.0, 0, 0])
        return self.grasp_pose(Rq @ self.handle_door - back * app + [0, 0, up], app, clo)

    def tuck_q(self):
        if self.q_tuck is None:
            downs = [DOWN_Q, qmult(np.array([0.0, 0, 0, 1]), DOWN_Q)] if self.try_flip else [DOWN_Q]
            self.q_tuck = [self.arm.ik(sapien.Pose(M.mw_to_world(TUCK_MW), d), np.r_[NOMINAL_Q, 0.04, 0.04],
                                       prefer_q=NOMINAL_Q) for d in downs]
        # --smooth: the hand down, either way round about the vertical (symmetric fingers): the closer
        qs = [q for q in self.q_tuck if q is not None]
        return min(qs, key=lambda q: np.abs(q - self.arm.q_cmd).sum()) if qs else None

    def ready(self):
        # a ready pose with the hand already level and facing the microwave, reached in joint
        # space: turning the hand level along a Cartesian path runs joint 4 into its limit
        # --smooth: the handle grasp is chosen first (plan_handle) and the ready pose is solved
        # from its approach, on the same wrist branch: the joint move to the handle is then short
        arm = self.arm
        P = self.grasp_pose(READY_MW, [0, 1.0, 0], [1.0, 0, 0])
        q_ready = None
        if self.handle_plan is not None and len(self.handle_plan[2]) == 10:
            # back from the handle's approach to the ready pose along a line, IK seeded step by
            # step (the same orientation throughout): the ready arm on the grasp's own branch
            P0 = self.handle_pose(self.handle_q0, back=0.10)
            n = int(np.ceil(np.linalg.norm(P.p - P0.p) / 0.01))
            qs = arm.path([sapien.Pose(P0.p + (P.p - P0.p) * k / n, P.q) for k in range(1, n + 1)],
                          self.handle_plan[2][-1])
            # the grasp holds j5 near its limit, and the line may run into it before the ready
            # pose: then ready is as far along it as the branch goes
            q_ready = qs[-1] if qs else None
            print(f"ready from the handle approach: {len(qs)}/{n} cm", flush=True)
        if q_ready is None:
            q_ready = arm.ik(P, np.r_[NOMINAL_Q, 0.04, 0.04], prefer_q=NOMINAL_Q)
        return self.step("ready", q_ready is not None and arm.move_joint(q_ready))

    def plan_handle(self, open_deg):
        """The arm configuration to hold the handle with, chosen so that the whole arc can be
        followed: from many IK solutions at the closed handle, continue each along the arc and
        keep the one that gets furthest (ties: furthest from the joint limits; --smooth: then
        the closest to the arm). Sets self.handle_plan = (q_grasp, arc, approach)."""
        arm, handle_pose = self.arm, self.handle_pose
        q_open = np.deg2rad(open_deg)
        # from the door's angle now: shut, or left part-open (a start in the middle of the task)
        q0 = float(self.u.door_angle()[0])
        self.handle_q0 = q0 = 0.0 if q0 < np.deg2rad(2) else q0
        best = None
        for fl in self.flips():
            self.flip = fl
            arc = [handle_pose(q) for q in np.arange(q0 + np.deg2rad(1), q_open + 1e-6, np.deg2rad(1))]
            for qc in arm.candidates(handle_pose(q0)):
                qs = arm.path(arc, qc)
                m = arm.margin([qc] + qs)
                key = (len(qs), m > 0.05, -np.abs(qc - arm.q_cmd).sum()) if self.try_flip else (len(qs), m)
                if best is None or key > best[0]:
                    best = (key, qc, qs, fl, m)
        self.flip = best[3]
        print(f"handle grasp: arc followed to {len(best[2])} deg, joint-limit margin {best[4]:.2f} rad, "
              f"flip {self.flip}", flush=True)
        q_grasp, arc_qs = best[1], best[2]
        self.handle_margin = float(best[4])
        approach = arm.path([handle_pose(q0, back=b) for b in np.arange(0.01, 0.1001, 0.01)], q_grasp)
        self.handle_plan = (q_grasp, arc_qs, approach)
        return len(approach) == 10

    def open_door(self, open_deg, miss_back=0.0, miss_deg=30):
        """miss_back > 0: a failed first attempt before the real one -- the hand closes miss_back short
        of the bar, makes the first miss_deg of the pulling arc (the door does not move), opens, backs
        out along the approach and grasps again (data in which the door's state decides what comes next)"""
        arm, step, handle_pose = self.arm, self.step, self.handle_pose
        if self.handle_plan is None:
            self.plan_handle(open_deg)
        q_grasp, arc_qs, approach = self.handle_plan
        self.handle_plan = None
        arm.gripper(HALF_OPEN)
        miss_back = miss_back if self.handle_q0 == 0 else 0.0     # the missed grasp only on a shut door
        ok = step("reach handle", len(approach) == 10 and arm.move_joint(approach[-1]))
        if ok and miss_back > 0:
            k = int(round(miss_back / 0.01)) - 1                 # approach[k]: the pose miss_back off the bar
            ok = step("reach short of the bar (miss)", arm.execute(approach[::-1][1:len(approach) - k]))
            arm.gripper(CLOSED)
            miss_qs = arm.path([handle_pose(q, back=miss_back) for q in np.deg2rad(np.arange(1, miss_deg + 1))],
                               approach[k])
            ok = ok and step(f"pull on nothing ({len(miss_qs)} deg)", len(miss_qs) > 0 and arm.execute(miss_qs, settle=4))
            ok = ok and step("door still shut", float(self.u.door_angle()[0]) < np.deg2rad(3))
            arm.gripper(HALF_OPEN)
            ok = ok and step("back out, retry", arm.execute(miss_qs[::-1][1:] + [approach[k]] + list(approach[k + 1:])))
        ok = ok and step("grasp handle", arm.execute(approach[::-1][1:] + [q_grasp]))
        arm.gripper(CLOSED)
        ok = ok and step(f"open to {len(arc_qs)} deg", len(arc_qs) > 0 and arm.execute(arc_qs, settle=4))
        arm.gripper(OPEN)
        q_now = float(self.u.door_angle()[0])
        ok = ok and step("release, back off", arm.go([handle_pose(q_now, back=0.10)]))
        ok = ok and step("up above the door", arm.go([handle_pose(q_now, back=0.10, up=0.23)]))
        # the hand is now outside the open door (beyond its face, at the handle); get round the
        # door's free edge (y = -0.356 at 90 deg) before crossing over, or the forearm sweeps
        # through the door and swings it shut by ~20 deg
        hp = M.world_to_mw(arm.tcp().p)
        ok = ok and step("round the door's edge",
                         arm.go([self.grasp_pose([hp[0], -0.55, hp[2]], *self.door_dirs(q_now))], high=True))
        ok = ok and step("across, in front",
                         arm.go([self.grasp_pose([0.15, -0.55, hp[2]], *self.door_dirs(q_now))], high=True))
        # crossings from one side of the microwave to the other go through a tucked pose near
        # the robot: joint-space moves out of it are short, and stay clear of the open door
        q_tuck = self.tuck_q()
        return ok and step("tuck", q_tuck is not None and arm.move_joint(q_tuck, step=0.02))

    # Each phase below is planned as a whole: IK candidates at its first pose, continued
    # through all of it; the arm gets to the first pose by a joint-space move made where
    # nothing is in the way (above the microwave and the door, or out in front).

    def pick_and_place(self, place, from_tcp=False, hold=False, human=None, human_smooth=False, human_carry_only=False,
                       carry_z=0.25, short_y=None, slide=False, miss_short=0.0):
        """2+3. Pick the pot and put it in the cavity, planned as ONE pose sequence --
        approach, grasp, lift, across high, down in front of the opening, in, down -- so the
        whole carry is a single continuous joint path: any joint-space move while holding the
        pot turns it in the grip (then it goes in tilted, its rim catches a finger, and backing
        out drags it along). from_tcp: the approach starts where the hand is now (it is
        already in front of the pot, on the approach line) instead of 10 cm in front of it.
        hold: the hand has already closed on the pot; carry it from where it is.
        short_y: a failed place -- the pot is let go at the opening, its centre at short_y (mw), half on the
        cavity's front edge (not 'inside': that needs y > 0.03), instead of at place.
        slide: the pot is at the opening (after short_y): grasp it from the front where it is and slide it
        in, level, to place -- no lift (a lift from the opening hits the top of the cavity).
        miss_short > 0: a failed grasp first -- the hand closes miss_short (m) short of the grasp on the approach
        line (nothing between the fingers), lifts 3 cm, finds it empty, opens, goes back down and on to the
        real grasp. Only on the solver's own approach (not from_tcp / hold / slide).
        human: sim/human_paths.load(take) -- follow the demonstrator's hand where it was seen:
        its approach (moved with the pot, kept in front of and above it) and its carry path
        (warped from the take's pick/place to this episode's), with this solver's hand
        orientation; the last 10 cm to the grasp are aligned to the pot's true pose and the
        insertion (unseen in the video) is planned here. human_smooth: fit the hand paths with
        a smoothing spline (human_paths.smooth_path) and join them to the solver's segments
        without corners, instead of following the tracked points as they are."""
        arm, step, u, der = self.arm, self.step, self.u, self.der
        smooth = human_smooth
        p = u.pot_in_mw().cpu().numpy()[0]
        # pot_z: started on a raised surface; a pot that has been moved is taken at its height now
        c = np.array([p[0], p[1], POT_GRASP_Z + (p[2] if abs(p[2] - u.pot_z) > 0.01 else u.pot_z)])
        # level, from the front (+y_mw), fingers across the pot's short side: the same hand
        # orientation as for going into the cavity, so the carry is a pure translation
        app, clo = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
        x, y = place
        z_place = der["turntable_top_z"] + POT_GRASP_Z + 0.001   # 1 mm drop: no tilt on release
        if short_y is not None:
            # (put down on the counter in front of the microwave instead: the arm's wrist links reach the
            # counter before the hand does there, close to the base, and the pot is dropped from 3 cm)
            y = short_y
        g = lambda xyz: self.grasp_pose(xyz, app, clo)
        if hold:
            c = M.world_to_mw(arm.tcp().p)
            seg_reach = [g(c)]
        elif from_tcp:
            seg_reach = arm.line(g(c))
        else:
            seg_reach = [g(c - (0.10 - d) * app) for d in np.arange(0, 0.1001, 0.01)]
        def reach0_now():                                       # with the current self.flip
            if hold:
                return [g(c)]
            if from_tcp:
                return arm.line(g(c))
            return [g(c - (0.10 - d) * app) for d in np.arange(0, 0.1001, 0.01)]

        def build(use_app, use_carry):
            """The pose sequence; the human's approach and/or carry path where asked."""
            seg_reach = reach0_now()
            self.human_target = {}                             # the human points actually commanded
            if use_app:
                ap = human["approach"] + np.r_[c[:2], 0]
                ap[:, 1] = np.minimum(ap[:, 1], c[1] - 0.09)       # stay in front of the pot ...
                ap[:, 2] = np.maximum(ap[:, 2], POT_GRASP_Z)        # ... and not below grasp height
                if smooth:                                          # ... smoothly into the pre-grasp pose
                    ap = HP.smooth_path(np.vstack([ap, c - 0.10 * app]), fix_end=True)
                    ap[:, 1] = np.minimum(ap[:, 1], c[1] - 0.09)
                    ap[:, 2] = np.maximum(ap[:, 2], POT_GRASP_Z)
                seg_app = [g(pt) for pt in ap]
                self.human_target["approach"] = np.round(ap, 4).tolist()
                seg_reach = seg_app + arm.line(g(c - 0.10 * app), start=seg_app[-1]) \
                    + [g(c - (0.10 - d) * app) for d in np.arange(0.01, 0.1001, 0.01)]
            if use_carry:
                cp = HP.warp_carry(human["carry"], human["pick"], human["place"], c[:2], place)
                cp[:, 2] = np.maximum(cp[:, 2], POT_GRASP_Z + 0.06)  # the pot clear of the counter
                cp[:, 1] = np.minimum(cp[:, 1], -0.06)               # in front of the microwave
                seg_lift = arm.line(g([c[0], c[1], cp[0, 2]]), start=seg_reach[-1])
                down_end = np.array([x, -0.10, z_place + 0.02])
                if smooth:
                    # one smooth curve from the top of the lift through the hand's carry to the
                    # point in front of the opening: no corners where solver and human meet
                    cp = HP.smooth_path(np.vstack([[c[0], c[1], cp[0, 2]], cp, down_end]),
                                        fix_start=True, fix_end=True)[1:]
                    seg_across = [g(pt) for pt in cp]
                else:
                    seg_across = arm.line(g(cp[0]), start=seg_lift[-1]) + [g(pt) for pt in cp[1:]]
                self.human_target["carry"] = np.round(cp, 4).tolist()
                seg_down = arm.line(g(down_end), start=seg_across[-1])
            else:
                # carry_z: the tcp height of the solver's own carry (0.25 = 20 cm above the grasp)
                seg_lift = arm.line(g([c[0], c[1], carry_z]), start=seg_reach[-1])
                seg_across = arm.line(g([x, -0.30, carry_z]), start=seg_lift[-1])
                seg_down = arm.line(g([x, -0.30, z_place + 0.02]), start=seg_across[-1])
            if slide:
                seg_lift, seg_across, seg_down = [], [], []
                seg_in = arm.line(g([x, y, c[2] + 0.005]), start=seg_reach[-1]) + [g([x, y, z_place])]
            else:
                seg_in = arm.line(g([x, y, z_place + 0.02]), start=seg_down[-1]) + [g([x, y, z_place])]
            return [seg_reach, seg_lift, seg_across, seg_down, seg_in]

        # with a human path: the whole of it, else only its carry, else none -- whichever the
        # arm can follow as one continuous joint path (planning moves nothing)
        modes = ([(False, True)] if human_carry_only else [(True, True)]) + [(False, True), (False, False)] \
            if (human is not None and not (from_tcp or hold)) \
            else [(False, False)]
        for use_app, use_carry in modes:
            cand = []
            for fl in self.flips():
                self.flip = fl
                segs = build(use_app, use_carry)
                if from_tcp or hold:
                    qs = arm.path(sum(segs, []), arm.q_cmd)            # continue from the arm as it is
                else:
                    qs = arm.plan(sum(segs, []))
                n_all = sum(len(sg) for sg in segs)
                cand.append(((len(qs) == n_all, -np.abs(qs[0] - arm.q_cmd).sum() if qs else -1e9), fl, segs, qs))
            (complete, _), self.flip, segs, qs = max(cand, key=lambda t: t[0])
            if complete:
                break
        self.human_parts = dict(approach=bool(use_app), carry=bool(use_carry)) if human is not None else None
        seg_reach = segs[0]
        n_all = sum(len(sg) for sg in segs)
        ok = step(f"plan pick-and-place ({len(qs)}/{n_all})"
                  + (f", human path: approach {use_app}, carry {use_carry}" if human is not None else ""), len(qs) == n_all)
        if ok:
            k = np.cumsum([0] + [len(sg) for sg in segs])
            if from_tcp or hold:
                ok = step("reach, grasp", arm.execute(qs[0:k[1]]))
            else:
                qv = arm.ik(g(M.world_to_mw(seg_reach[0].p) + [0, 0, 0.15]) if use_app else g(c - 0.10 * app + [0, 0, 0.30]),
                            np.r_[qs[0], 0.04, 0.04])
                if qv is not None:
                    arm.move_joint(qv, step=0.02)                  # over the pot, high
                arm.move_joint(qs[0], step=0.02)
                m = int(round(miss_short / 0.01)) if (miss_short > 0 and not slide) else 0
                if 0 < m < k[1] - 1:
                    ok = step("reach short of the pot (miss)", arm.execute(qs[1:k[1] - m]))
                    arm.gripper(CLOSED)
                    short = M.world_to_mw(arm.tcp().p)
                    up = arm.path([g(short + [0, 0, dz]) for dz in np.arange(0.01, 0.0301, 0.01)], arm.q_cmd)
                    ok = ok and step("lift on nothing (miss)", len(up) == 3 and arm.execute(up, settle=4))
                    ok = ok and step("pot still down", float(u.pot_in_mw()[0, 2].cpu()) < float(p[2]) + 0.01)
                    arm.gripper(OPEN, n=30)
                    ok = ok and step("down, retry", arm.execute(up[::-1][1:] + [qs[k[1] - m - 1]]))
                    ok = ok and step("reach, grasp", arm.execute(qs[k[1] - m:k[1]]))
                else:
                    ok = step("reach, grasp", arm.execute(qs[1:k[1]]))
            arm.gripper(CLOSED)
            names = ("lift", "carry (human path)", "down to the opening", "in, down") if use_carry else \
                ("lift", "across, high", "down", "in, down")
            for name, i0, i1 in zip(names, range(1, 5), range(2, 6)):
                if k[i1] > k[i0]:
                    ok = ok and step("slide in" if slide else name, arm.execute(qs[k[i0]:k[i1]], settle=4))
        if ok:
            arm.gripper(OPEN, n=30)
            if os.environ.get("DEBUG_RELEASE"):
                for f in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
                    link = [l for l in u.agent.robot.links if l.name.endswith(f)][0]
                    print("   ", f, "force on pot",
                          np.round(u.scene.get_pairwise_contact_forces(link, u.pot).cpu().numpy()[0], 2))
            # the fingers sit just under the lid's rim (74 mm, 3 mm inside the open fingers):
            # drop 8 mm clear of it before backing out
            ok = step("clear the rim", arm.go([g([x, y, z_place - 0.008])]))
            # back the way it came in (that path was planned as one continuous IK path, so
            # its reverse is feasible): out to where it came down, then up that line
            ok = ok and step("out", arm.go([g([x, -0.30, z_place - 0.008])]))
            ok = ok and step("up", arm.go([g([x, -0.30, 0.25])], high=True))
            ok = ok and step("tuck", arm.move_joint(self.tuck_q(), step=0.02))
        return ok

    def shut_door(self):
        """4. Hold the handle, carry it round towards 0 as far as one arm configuration
        allows, let go, come down onto the bar again from above in another configuration,
        continue. From above works wherever the door ended up (the arm may have pushed it
        to 100 deg)."""
        arm, step, handle_pose = self.arm, self.step, self.handle_pose
        ok = True
        for regrasp in range(3):
            q_now = float(self.u.door_angle()[0])
            if q_now < np.deg2rad(2):
                break
            best, bkey = None, None
            for fl in self.flips():
                self.flip = fl
                back_arc = [handle_pose(q) for q in np.arange(q_now - np.deg2rad(1), -np.deg2rad(0.5), -np.deg2rad(1))]
                above = [handle_pose(q_now, up=h) for h in np.arange(0.01, 0.1001, 0.01)]
                for qc in arm.candidates(handle_pose(q_now)):
                    appr = arm.path(above, qc)
                    if len(appr) < len(above):
                        continue
                    qs = arm.path(back_arc, qc)
                    key = (len(qs), -np.abs(appr[-1] - arm.q_cmd).sum()) if self.try_flip else (len(qs),)
                    if best is None or key > bkey:
                        best, bkey, best_fl = (qc, appr, qs), key, fl
            if best is not None:
                self.flip = best_fl
            ok = step(f"plan shut from {np.degrees(q_now):.0f} deg"
                      + (f" ({len(best[2])}/{len(back_arc)})" if best else " (no approach)"),
                      best is not None and len(best[2]) > 0)
            if not ok:
                break
            q_hold, appr, shut_qs = best
            qv = arm.ik(handle_pose(q_now, up=0.20), np.r_[appr[-1], 0.04, 0.04])
            if qv is not None:
                arm.move_joint(qv, step=0.02)
            arm.move_joint(appr[-1], step=0.02)
            arm.gripper(HALF_OPEN)
            ok = step("down onto the handle", arm.execute(appr[::-1][1:] + [q_hold]))
            arm.gripper(CLOSED)
            ok = ok and step("shut", arm.execute(shut_qs, settle=10))
            arm.gripper(HALF_OPEN)
            q_end = float(self.u.door_angle()[0])
            ok = ok and step("up off the handle", arm.go([handle_pose(q_end, up=0.12)], high=True))
            if not ok:
                break
        return ok

    def result(self):
        info = self.u.evaluate()
        return dict(success=bool(info["success"][0]), pot_inside=bool(info["pot_inside"][0]),
                    door_deg=float(info["door_deg"][0]), phases=self.log,
                    pot_mw=np.round(self.u.pot_in_mw().cpu().numpy()[0], 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--open-deg", type=float, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pot", type=float, nargs=2, default=(0.50, -0.25), help="pot x y in the mw frame")
    ap.add_argument("--place", type=float, nargs=2, default=PLACE_MW, help="place x y in the mw frame")
    ap.add_argument("--take", help="read --pot and --place from recon/takes/<take>/pot.json")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dart-noise", type=float, default=0.0, help="std (rad) of the executed-target noise")
    ap.add_argument("--human", help="follow this take's hand paths (sim/human_paths.py), e.g. take_000")
    ap.add_argument("--human-smooth", action="store_true", help="--human: smoothing-spline the hand paths, no corners")
    ap.add_argument("--human-carry-only", action="store_true", help="--human: the hand's carry only; the solver's own approach")
    ap.add_argument("--pot-z", type=float, default=0.0, help="pot start height (m), e.g. 0.277 = on the microwave")
    ap.add_argument("--carry-z", type=float, default=0.25, help="tcp height of the solver's own carry (m, mw frame)")
    ap.add_argument("--speed", type=float, default=1.0, help="arm speed factor (gripper timing unchanged)")
    ap.add_argument("--smooth", action="store_true", help="fewer wrist flips: the 180-deg-turned grasp allowed, and "
                    "among complete plans the IK solution closest to the arm")
    ap.add_argument("--j5-max", type=float, default=0.0, help="--smooth: also keep |j5| below this (deg); 0: off "
                    "(the handle grasp needs j5 near its limit)")
    ap.add_argument("--miss-back", type=float, default=0.0, help="> 0: a missed handle grasp (this far short of the bar, m) and a retry")
    ap.add_argument("--pot-miss", type=float, default=0.0, help="> 0: a missed pot grasp (the hand closes this far short "
                    "of the grasp, m), a 3 cm lift on nothing, and a retry")
    ap.add_argument("--base", type=float, nargs=2, default=None, help="the Panda's base in the mw frame "
                    "(default microwave_env.ROBOT_IN_MW, (0.25, -0.70))")
    ap.add_argument("--start", choices=["closed", "half", "open", "inside", "lip"], default="closed",
                    help="the task's state at the start: door shut (the whole task); door part-open (re-open it, "
                    "then the rest); door open with the pot on the counter / in the cavity / just in front of the "
                    "opening (--pot); the solver does what is left")
    ap.add_argument("--door-deg", type=float, default=90.0, help="--start other than closed: the door's start angle")
    ap.add_argument("--place-short", type=float, default=None, help="a failed place first: the pot let go with its "
                    "centre at this y (mw; e.g. 0.015, on the cavity's front edge, not 'inside'), then grasped where it is "
                    "and slid in to --place")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.take:
        tk = json.load(open(os.path.join(ROOT, "Gitignore-but-important/recon/takes", a.take, "pot.json")))
        a.pot, a.place = (tk["pick"]["x"], tk["pick"]["y"]), (tk["end"]["x"], tk["end"]["y"])
        print(f"{a.take}: pot from {np.round(a.pot, 3)}, place at {np.round(a.place, 3)}", flush=True)

    # the pot where the takes picked it up, its short side facing the front (the Panda's
    # 80 mm opening fits the 48-55 mm short side, not the 80 mm long one)
    pot_xy = np.array(a.pot)
    yaw = np.pi / 2                    # long side along +y_mw: grasped from the front, carried in square
    # --no-video: no renderer at all (physics only; also runs where there is no Vulkan)
    render = {} if a.no_video else dict(render_mode="rgb_array")
    if a.start == "inside":                     # the pot on the turntable
        a.pot_z = json.load(open(M.MW_PARAMS))["derived"]["turntable_top_z"]
    door_q0 = 0.0 if a.start == "closed" else np.deg2rad(a.door_deg)
    if a.base is not None:
        M.set_robot_in_mw(a.base)
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", pot_mw=(*pot_xy, yaw),
                   pot_z=a.pot_z, door_q0=door_q0, sim_backend="cpu", max_episode_steps=4000,
                   **(render or dict(render_backend="none")))
    env = RecordEpisode(env, a.out, save_video=not a.no_video, save_trajectory=True, trajectory_name="solve",
                        video_fps=30, max_steps_per_video=4000)
    obs0, _ = env.reset(seed=a.seed)
    arm = Arm(env, obs=obs0, noise=a.dart_noise, rng_seed=(a.seed + int(1e4 * (a.pot[0] + a.pot[1]))) % 2**32, speed=a.speed,
              j5_max=np.deg2rad(a.j5_max) if a.smooth and a.j5_max else None)
    arm.closest_first = a.smooth
    S = Solver(env, arm)
    S.try_flip = a.smooth
    if a.start in ("closed", "half"):
        if a.smooth:
            S.plan_handle(a.open_deg)
        if a.start == "closed":
            ok = S.ready()
        else:                                   # the ready pose is where a part-open door is: go high instead
            q_tuck = S.tuck_q()
            ok = S.step("tuck", q_tuck is not None and arm.move_joint(q_tuck, step=0.02))
        ok = S.open_door(a.open_deg, miss_back=a.miss_back) and ok
    else:                                       # the door is open already: out of the way, then the rest
        q_tuck = S.tuck_q()
        ok = S.step("tuck", q_tuck is not None and arm.move_joint(q_tuck, step=0.02))
    slide = False
    if ok and a.place_short is not None and a.start != "inside":
        ok = S.pick_and_place(a.place, carry_z=a.carry_z, short_y=a.place_short)
        ok = S.step("left at the opening (not inside)", ok and not bool(S.u.evaluate()["pot_inside"][0]))
        slide = True
    if ok and a.start != "inside":
        ok = S.pick_and_place(a.place, human=HP.load(a.human) if a.human and not slide else None,
                              human_smooth=a.human_smooth, human_carry_only=a.human_carry_only, carry_z=a.carry_z,
                              slide=slide, miss_short=a.pot_miss)
    if ok:
        ok = S.shut_door()

    res = S.result()
    res["all_planned"] = ok
    res["human"] = a.human
    res["human_parts"] = getattr(S, "human_parts", None)
    res["human_target"] = getattr(S, "human_target", None)
    res["human_smooth"] = bool(a.human_smooth)
    res["human_carry_only"] = bool(a.human_carry_only)
    res["carry_z"] = a.carry_z
    res["speed"] = a.speed
    res["miss_back"] = a.miss_back
    res["pot_miss"] = a.pot_miss
    res["joint_jumps"] = arm.jumps
    res["smooth"] = bool(a.smooth)
    res["start"], res["door_deg0"], res["place_short"] = a.start, float(np.degrees(door_q0)), a.place_short
    res["base"] = M.ROBOT_IN_MW.tolist()
    res["handle_margin_rad"] = getattr(S, "handle_margin", None)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    # the (observation, clean planned action) pairs; with --dart-noise these differ from
    # what was executed, which is the point
    np.savez_compressed(os.path.join(a.out, "demo.npz"), obs=np.array(arm.rec_obs, np.float32),
                        actions=np.array(arm.rec_act, np.float32), dart_noise=a.dart_noise)
    print("RESULT", {k: v for k, v in res.items() if k != "phases"})
    env.close()


if __name__ == "__main__":
    main()
