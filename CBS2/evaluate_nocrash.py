from runners import NoCrashEvalRunner

def main(args):

    if(args.mod == 'fpn'):
        args.agent_config = 'results/config_nocrash_fpn.yaml'
    elif(args.mod == 'ppm'):
        args.agent_config = 'results/config_nocrash_ppm.yaml'
    elif(args.mod == 'lbc'):
        args.agent_config = 'results/config_nocrash_lbc.yaml'
    elif(args.mod == 'worcbs2'):
        args.agent_config = 'results/config_nocrash_wor_cbs2.yaml'
    else:
        args.agent_config = 'results/config_nocrash_original.yaml'

    town = args.town
    weather = args.weather

    port = args.port
    tm_port = port + 2
    runner = NoCrashEvalRunner(args, town, weather, port=port, tm_port=tm_port)
    runner.run()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # Agent configs
    #parser.add_argument('--agent', default='autoagents/image_agent')
    #parser.add_argument('--agent', default='autoagents/lbc_agent')
    #parser.add_argument('--agent', default='autoagents/cbs_agent')
    parser.add_argument('--agent', default='autoagents/cbs2_agent')
    #parser.add_argument('--agent-config', default='results/config_nocrash_cbs2.yaml')
    parser.add_argument("--mod", type=str,
                        default='original',
                        help="Perception module (original, ppm, fpn)")

    # Benchmark configs
    parser.add_argument('--town', required=True, choices=['Town01', 'Town02'])
    parser.add_argument('--weather', required=True, choices=['train', 'test'])

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    args = parser.parse_args()

    main(args)
