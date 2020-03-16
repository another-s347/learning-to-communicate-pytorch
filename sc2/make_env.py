from pysc2 import maps
from pysc2.env import sc2_env, sc2_env_test
from absl import flags
from pysc2.lib import point_flag

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "48",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "48",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", "RAW", sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", True,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", True,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", True, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 4, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string(
    "agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "1", "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")


def make_env(opt):
    map_inst = maps.get(opt.map)

    players = []

    agent_module, agent_name = opt.agent.rsplit(".", 1)

    players.append(sc2_env.Agent(sc2_env.Race[opt.agent_race],
                                 opt.agent_name or agent_name))

    if map_inst.players >= 2:
        if opt.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[opt.agent2_race],
                                       sc2_env.Difficulty[opt.difficulty],
                                       sc2_env.BotBuild[opt.bot_build]))
        else:
            agent_module, agent_name = opt.agent2.rsplit(".", 1)
            players.append(sc2_env.Agent(sc2_env.Race[opt.agent2_race],
                                         opt.agent2_name or agent_name))

    env = sc2_env.SC2Env(
        map_name=opt.map,
        battle_net_map=opt.battle_net_map,
        players=players,
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=opt.feature_screen_size,
            feature_minimap=opt.feature_minimap_size,
            rgb_screen=opt.rgb_screen_size,
            rgb_minimap=opt.rgb_minimap_size,
            action_space=opt.action_space,
            use_feature_units=opt.use_feature_units,
            use_raw_units=opt.use_raw_units,
            use_camera_position=True,
            send_observation_proto=False,
            camera_width_world_units=48),
        step_mul=opt.step_mul,
        game_steps_per_episode=4000,
        disable_fog=opt.disable_fog,
        visualize=True,
        ensure_available_actions=True)

    return env