# Powderworld

Powderworld is a lightweight simulation environment for understanding AI generalization.

As the AI research community moves towards general agents, which can solve many tasks within an environment, it is crucial that we study environments with sufficient complexity. An ideal "foundation environment" should provide rich dynamics that stand in for real-world physics, while remaining fast to evaluate and reset.

To this end, we introduce Powderworld, a lightweight simulation environment for understanding AI generalization. Powderworld presents a 2D ruleset where elements (e.g. sand, water, fire) modularly react within local neighborhoods. Local interactions combine to form wide-scale emergent phenomena, allowing diverse tasks to be defined over the same core rules. Powderworld runs directly on the GPU, allowing up to 10,000 simulation steps per second, an order of magnitude faster than other RL environments like Atari or MineRL.
