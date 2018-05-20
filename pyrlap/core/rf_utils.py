def toStateActionRFdict(rf, tf):
	if type(list(rf.values())[0]) == dict:
		if type(list(rf.values())[0].values()[0]) == dict:
			is_state_action_nextstateRF = True
		else:
			is_state_action_nextstateRF = False
		is_stateRF = False
	else:
		is_stateRF = True

	temp_rf = {}
	stateactions = dict([(s, sorted(a.keys())) for s, a in tf.items()])
	for s, acts in stateactions.items():
		temp_rf[s] = {}
		for a in acts:
			ns = tf[s][a]
			if type(ns) == dict:
				ns = list(ns.keys())[0]
			else:
				raise Exception("Error: multiple next states "
								"- cannot assign reward")

			if is_stateRF:
				temp_rf[s][a] = rf[ns]
			elif is_state_action_nextstateRF:
				temp_rf[s][a] = rf[s][a][ns]
			else:
				temp_rf[s][a] = rf[s][a]
	return temp_rf

def toStateActionNextstateRFdict(rf, tf):
	if type(list(rf.values())[0]) == dict:
		is_stateRF = False
	else:
		is_stateRF = True

	temp_rf = {}
	for s, a_ns_prob in tf.items():
		temp_rf[s] = {}
		for a, ns_prob in a_ns_prob.items():
			temp_rf[s][a] = {}
			for ns, prob in ns_prob.items():
				if is_stateRF:
					temp_rf[s][a][ns] = rf[ns]
				else:
					temp_rf[s][a][ns] = rf[s][a]
	return temp_rf