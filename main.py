import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path

from utils.dotdic import DotDic
from arena import Arena
from agent import CNetAgent
from switch.switch_game import SwitchGame
from switch.switch_cnet import SwitchCNet
from sc2.sc2_game import Sc2Game
from sc2.sc2_cnet import Sc2CNet
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path',None,'path to existing options file')
flags.DEFINE_string('results_path',None,'path to results directory')
flags.DEFINE_integer('ntrials',1,'number of trials to run')
flags.DEFINE_integer('start_index',0,'starting index for trial output')
flags.DEFINE_bool('verbose', True, 'prints training epoch rewards if set')

"""
Play communication games
"""

# configure opts for Switch game with 3 DIAL agents
def init_action_and_comm_bits(opt):
	opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1
	if opt.model_comm_narrow is None:
		opt.model_comm_narrow = opt.model_dial
	if not opt.model_comm_narrow and opt.game_comm_bits > 0:
		opt.game_comm_bits = 2 ** opt.game_comm_bits
	if opt.comm_enabled:
		opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
	else:
		opt.game_action_space_total = opt.game_action_space
	return opt

def init_opt(opt):
	if not opt.model_rnn_layers:
		opt.model_rnn_layers = 2
	if opt.model_avg_q is None:
		opt.model_avg_q = True
	if opt.eps_decay is None:
		opt.eps_decay = 1.0
	opt = init_action_and_comm_bits(opt)
	return opt

def create_game(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchGame(opt)
	elif game_name == 'sc2':
		return Sc2Game(opt)
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_cnet(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchCNet(opt)
	elif game_name == 'sc2':
		return Sc2CNet(opt)
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_agents(opt, game):
	agents = [None] # 1-index agents
	cnet = create_cnet(opt)
	cnet_target = copy.deepcopy(cnet)
	for i in range(1, opt.game_nagents + 1):
		agents.append(CNetAgent(opt, game=game, model=cnet, target=cnet_target, index=i))
		if not opt.model_know_share:
			cnet = create_cnet(opt)
			cnet_target = copy.deepcopy(cnet)
	return agents

def save_episode_and_reward_to_csv(file, writer, e, r):
	writer.writerow({'episode': e, 'reward': r})
	file.flush()

def run_trial(opt, result_path=None, verbose=False):
	# Initialize action and comm bit settings
	opt = init_opt(opt)

	game = create_game(opt)
	agents = create_agents(opt, game)
	arena = Arena(opt, game)

	test_callback = None
	if result_path:
		result_out = open(result_path, 'w')
		csv_meta = '#' + json.dumps(opt) + '\n'
		result_out.write(csv_meta)
		writer = csv.DictWriter(result_out, fieldnames=['episode', 'reward'])
		writer.writeheader()
		test_callback = partial(save_episode_and_reward_to_csv, result_out, writer)
	arena.train(agents, verbose=verbose, test_callback=test_callback)

	if result_path:
		result_out.close()

def main(unused_arg):
	opt = DotDic(json.loads(open(FLAGS.config_path, 'r').read()))

	result_path = None
	if FLAGS.results_path:
		result_path = FLAGS.config_path and os.path.join(FLAGS.results_path, Path(FLAGS.config_path).stem) or \
			os.path.join(FLAGS.results_path, 'result-', datetime.datetime.now().isoformat())

	for i in range(FLAGS.ntrials):
		trial_result_path = None
		if result_path:
			trial_result_path = result_path + '_' + str(i + FLAGS.start_index) + '.csv'
		trial_opt = copy.deepcopy(opt)
		run_trial(trial_opt, result_path=trial_result_path, verbose=FLAGS.verbose)


if __name__ == '__main__':
	app.run(main)