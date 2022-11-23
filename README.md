# Powderworld

Powderworld is a lightweight simulation environment for understanding AI generalization.

As the AI research community moves towards general agents, which can solve many tasks within an environment, it is crucial that we study environments with sufficient complexity. An ideal "foundation environment" should provide rich dynamics that stand in for real-world physics, while remaining fast to evaluate and reset.

To this end, we introduce Powderworld, a lightweight simulation environment for understanding AI generalization. Powderworld presents a 2D ruleset where elements (e.g. sand, water, fire) modularly react within local neighborhoods. Local interactions combine to form wide-scale emergent phenomena, allowing diverse tasks to be defined over the same core rules. Powderworld runs directly on the GPU, allowing up to 10,000 simulation steps per second, an order of magnitude faster than other RL environments like Atari or MineRL.

View the [Project Website](https://kvfrans.com/static/powder/) for more details.



https://user-images.githubusercontent.com/1484166/203631565-bc5a4083-fa67-47a8-9e42-0f12e1eb2156.mp4



Examples:
```
# Train an agent on the Sand-Pushing env:
python examples/powder_agent.py --savedir test --env_name sand

# Train an 8-step world model.
python examples/powder_worldmodel.py --savedir test
```
