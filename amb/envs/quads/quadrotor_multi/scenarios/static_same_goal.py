from amb.envs.quads.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_static_same_goal(QuadrotorScenario):
    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        return
