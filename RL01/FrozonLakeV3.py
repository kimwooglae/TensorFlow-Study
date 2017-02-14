import gym
from gym.envs.registration import  register
import sys,tty,termios


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch



#Register FrozenLake with is_slippery false
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery': False}
)

LEFT = 0
DOWN =  1
RIGHT = 2
UP =  3

inkey = _Getch()

arraw_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}


env = gym.make("FrozenLake-v3")
env.render()

while True:
    key = inkey()
    if key not in arraw_keys.keys():
        print('Game aborted.')
        break

    action = arraw_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State:", state, " Action:", action, " Reward:", reward, " info:", info)

    if done:
        print("finished with reward:", reward)
        break
