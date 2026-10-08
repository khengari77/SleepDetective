# SleepDetective 
This is a project that aims to monitor vehicle drivers' vigilance and gives an alarm in case the driver loses his vigilance. Despite the name SleepDetective, the project monitors awareness in general. 
# TODOs
- [x] Design the structure of the project.
- [x] Build the system parts.
- [x] Test the system on hardware.
- [ ] Write tests for system.
- [ ] Decouple the deisgn by using dependency injection.
- [ ] Create No Hardware running mode.
- [ ] Replace Flask with [Flask-meld](https://www.flask-meld.dev/).
- [ ] Add an event logging system (maybe?)
# Project Structure
# How to install
Requires [uv](https://docs.astral.sh/uv/):
```bash
uv sync
uv run python main.py --help
```
# v2 (in progress)
A learned drowsiness estimator trained on UTA-RLDD is being built to replace
the heuristic stage — see `PLAN.md`. v2 code lives in `src/`, configuration in
`configs/`, and experiments write to `reports/`.
# Citation
The v2 model is trained on the UTA Real-Life Drowsiness Dataset:
```bibtex
@inproceedings{ghoddoosian2019realistic,
  title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
  author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```
