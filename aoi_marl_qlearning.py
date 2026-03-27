import random

random.seed(0)

# ===PARAMETERS===

NUM_SENSORS      = 5

# Actions
WAIT             = 0
TRANSMIT         = 1
ACTIONS          = [WAIT, TRANSMIT]

# Transmission status
IDLE             = 0
SUCCESS          = 1
COLLISION        = 2

# Q-learning
gamma            = 0.85
alpha            = 0.05
num_time_slots   = 100
num_episodes     = 5000

# Exploration (epsilon-greedy decay)
epsilon_start    = 1.0
epsilon_end      = 0.05
epsilon_decay    = (epsilon_end / epsilon_start) ** (1 / num_episodes)

# AoI
A_MAX            = NUM_SENSORS * 4

# Power
ENERGY_COST      = 2
POWER_BUDGET     = int((num_time_slots / NUM_SENSORS) * ENERGY_COST * 1.2)

# Evaluation
num_eval_episodes = 1000

class Sensor:
    def __init__(self, priority):
        self.aoi = 1
        self.priority = priority
        self.last_outcome = IDLE
        self.power_budget = POWER_BUDGET
        self.q_table = {}

    def get_reward(self):
        if self.last_outcome == SUCCESS:
            bonus = 0.5 * NUM_SENSORS * 2
        elif self.last_outcome == COLLISION:
            bonus = -2.5 * NUM_SENSORS * 2
        else: # IDLE
            bonus = -0.25 * NUM_SENSORS * 2

        aoi_penalty = -self.priority * min(self.aoi, A_MAX) ** 1.1
        
        return aoi_penalty + bonus
        
    def reset(self):
        self.aoi = 1

    def age(self):
        self.aoi += 1

    def get_state(self):
        return (min(self.aoi, A_MAX), self.last_outcome, self.power_budget // 20)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state, action, value):
        self.q_table[(state, action)] = value

    def choose_action(self, state, epsilon):
        available = [WAIT] if self.power_budget <= 0 else ACTIONS

        if random.random() < epsilon:
            return random.choice(available)

        q_vals = {a: self.get_q(state, a) for a in available}
        max_q = max(q_vals.values())

        best = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best)

class WirelessSensorNetwork:
    def __init__(self, num_sensors):
        self.num_sensors = num_sensors
        self.time = 0
        priorities = [1.5 if i < num_sensors // 2 else 1.0 for i in range(num_sensors)]
        self.sensors = [Sensor(priority=p) for p in priorities]
        self.past_rewards = []

    def step(self, actions):
        self.time += 1

        for i, sensor in enumerate(self.sensors):
            if actions[i] == TRANSMIT and sensor.power_budget <= 0:
                actions[i] = WAIT
        
        # s (before transition)
        states = [s.get_state() for s in self.sensors]

        transmitting = [i for i, a in enumerate(actions) if a == TRANSMIT]

        if len(transmitting) == 1:
            idx = transmitting[0]
            # transmitter succeeds
            self.sensors[idx].reset()
            self.sensors[idx].last_outcome = SUCCESS
            self.sensors[idx].power_budget -= ENERGY_COST

            for j, sensor in enumerate(self.sensors):
                if j != idx:
                    sensor.last_outcome = IDLE
        elif len(transmitting) == 0:
            for sensor in self.sensors:
                sensor.last_outcome = IDLE
        else:
            tx_set = set(transmitting)
            for j, sensor in enumerate(self.sensors):
                if j in tx_set:
                    sensor.last_outcome = COLLISION
                    sensor.power_budget = max(0, sensor.power_budget - ENERGY_COST)
                else:
                    sensor.last_outcome = IDLE

        # age sensors
        for j, sensor in enumerate(self.sensors):
            if not (len(transmitting) == 1 and j == transmitting[0]):
                sensor.age()

        # s' (after transition)
        next_states = [s.get_state() for s in self.sensors]

        step_reward = 0
        for sensor in self.sensors:
            step_reward += sensor.get_reward()

        # Q-learning update per sensor
        for i, sensor in enumerate(self.sensors):
            s = states[i]
            a = actions[i]
            r = sensor.get_reward()
            sp = next_states[i]

            current_q = sensor.get_q(s, a)
            max_next_q = max(sensor.get_q(sp, ap) for ap in ACTIONS)

            # Q(s,a) <- Q + alpha * (r + gamma * maxQ(s',a') - Q)
            new_q = current_q + alpha * (r + gamma * max_next_q - current_q)
            sensor.set_q(s, a, new_q)

        return step_reward

    def reset(self):
        self.time = 0
        for sensor in self.sensors:
            sensor.reset()
            sensor.last_outcome = IDLE
            sensor.power_budget = POWER_BUDGET

    def get_avg_aoi(self):
         return sum(sensor.aoi for sensor in self.sensors) / len(self.sensors)

    def get_cumulative_reward(self):
        cumulative_reward = 0
        for i in range(self.num_sensors):
            cumulative_reward += self.sensors[i].get_reward()

        return cumulative_reward

def random_policy(sensors):
    actions = []
    for sensor in sensors:
        if sensor.power_budget <= 0:
            actions.append(WAIT)
        else:
            actions.append(TRANSMIT if random.random() < 1 / len(sensors) else WAIT)
    return actions