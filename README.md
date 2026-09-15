# Agent Log Budget Study

Code accompanying the working manuscript **Failure Localization in Agent Trajectories under Limited Log Budgets: An Empirical Study**, by Hongming Guo.

This is an initial code release, not a claim of publication or a complete archival replication package. It compares full logs and prefix, suffix, and spread selections at 6,000 and 12,000 serialized Unicode characters. The three-provider experiment comprises 150 trajectories in 112 question groups, three repetitions, and 9,450 logical evaluations. The deployed model configurations differ; this is not a controlled model ranking.

## Environment and offline tests

The original execution environment was Windows with Python 3.12. The runner's process-lock check uses Windows `tasklist`.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

These tests make no API calls. Unit checks use synthetic inputs; the benchmark split integration check is skipped until upstream inputs are prepared.

## Prepare benchmark inputs

Obtain the upstream Who&When repository yourself, retaining its license:

```sh
git clone https://github.com/ag2ai/Agents_Failure_Attribution.git data/external/who_when/f4d2b6da464a826580e59b3a0eae15ea2d642d7c
git -C data/external/who_when/f4d2b6da464a826580e59b3a0eae15ea2d642d7c checkout f4d2b6da464a826580e59b3a0eae15ea2d642d7c
python scripts/prepare_log_study.py
```

Preparation writes inputs, evaluation labels, and group splits under `data/processed/log_study_v1`. Labels are used for scoring and are not included in the localization prompt. Group identifiers incorporate native path strings in this original script, so Windows is recommended to reproduce historical identifiers exactly. Cross-platform replication has not been validated.

## Paid model runs

Set `ZHIPUAI_API_KEY`, `DASHSCOPE_API_KEY`, and `MOONSHOT_API_KEY` locally. Never commit their values. Model IDs and generation settings are frozen in the runner; availability may change after the experiment.

```sh
python scripts/run_multivendor_formal.py --directory outputs/replication --limit 3 --workers 1 --interleave
```

This command makes real, billable API calls. Inspect the saved results before increasing the limit. Successful records are resumed automatically. `--retry-failed` retries failures, not valid incorrect predictions. Keep different configurations in separate directories. The historical output parser is `mv-3.1`.

## Offline analysis

After a complete run:

```sh
python scripts/analyze_multivendor.py --directory outputs/replication
```

Accuracy averages repetitions within trajectories, then trajectories within groups, then groups equally. Six comparisons are Holm-adjusted within each provider. The script's historical “first-pass” wording means no archived rerun cycle and can include transport retries; it does not mean first HTTP attempt. Cost output should not be treated as an invoice: usage-bearing failed archives must also be accounted for, and missing usage does not establish zero billing.

## Release scope

Raw benchmark data, original API responses, failed response archives, and credentials are not distributed in this initial release. Accordingly, the repository alone does not yet independently reproduce the paper's exact historical numbers. The manuscript and archival analysis package will be added after their release audit. No third-party data license is replaced by this repository.

## Reference and disclosure

Zhang et al. (2025), *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems*, ICML, PMLR 267:76583–76599. https://proceedings.mlr.press/v267/zhang25cq.html

The study is self-funded. The sole author declares no competing interests. AI tools assisted research planning, code development, analysis, and writing; the author is responsible for verification and the final work.
