phase_info = {
    'pre_mission': {
        'include_takeoff': True, 'optimize_mass': True}, 'climb_1': {
            'subsystem_options': {
                'core_aerodynamics': {
                    'method': 'computed'}}, 'user_options': {
                        'optimize_mach': True, 'optimize_altitude': True, 'polynomial_control_order': [
                            1, 2], 'use_polynomial_control': True, 'num_segments': [1], 'order': 1, 'solve_for_distance': True, 'initial_mach': (
                                1, None), 'final_mach': (
                                    2, None), 'mach_bounds': (
                                        (0.98, 2.02), None), 'initial_altitude': (
                                            1, None), 'final_altitude': (
                                                2, None), 'altitude_bounds': (
                                                    (0.0, 502), None), 'throttle_enforcement': 'path_constraint', 'fix_initial': True, 'constrain_final': True, 'fix_duration': False, 'initial_bounds': (
                                                        (0.0, 0.0), None), 'duration_bounds': (
                                                            (0.5, 1.5), None)}, 'initial_guesses': {
                                                                'time': (
                                                                    [
                                                                        1, 1], None)}}, 'post_mission': {
                                                                            'include_landing': True, 'constrain_range': True, 'target_range': (
                                                                                514.5, None)}}
