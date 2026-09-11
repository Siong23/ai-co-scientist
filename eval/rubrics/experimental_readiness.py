"""The Experimental readiness ladder: can this hypothesis be run as an experiment?

``GEval`` asks a judge for a holistic 0-1 opinion, which moves between runs and
cannot be audited. Experimental readiness is not a matter of opinion: a
hypothesis is executable exactly when it names the four things an experimenter
needs before touching any equipment.

    intervention  -> what gets manipulated (the independent variable)
    measurement   -> what gets recorded (the dependent variable)
    comparison    -> what the result is judged against (control or baseline)
    prediction    -> which way the effect is expected to go

So this metric is a ``DAGMetric``: a decision tree of yes/no judgements. Each
rung is decided independently, the score is fixed by which rung the hypothesis
falls off, and every lost point names the slot that was empty. "0.6 because it
states no comparator" is reviewable in a way that "0.6" is not.

The ladder is ordered by dependency, not by importance. A dependent variable
means nothing without an intervention to attribute it to, and a predicted
direction means nothing without something to measure, so a hypothesis that
fails an early rung cannot be rescued by a later one.

Scores are the DeepEval 0-10 verdict scale; ``DAGMetric`` divides by ten, so
the reported score is 0-1 like every other metric in the suite.
"""

from __future__ import annotations

from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph
from deepeval.test_case import SingleTurnParams

#: The judge reads the goal and the hypothesis, and nothing else. No rung may
#: consult retrieval context: an experimenter reads the hypothesis, and a slot
#: that only the evidence fills is still a slot the hypothesis left empty.
EVALUATION_PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

#: Rung scores, out of ten. The gap between "no intervention" and every other
#: failure is deliberate: without a manipulated variable there is no experiment
#: to design, whereas a missing comparator still leaves something to build on.
SCORE_NO_INTERVENTION = 0
SCORE_NO_MEASUREMENT = 4
SCORE_NO_COMPARISON = 6
SCORE_NO_PREDICTION = 8
SCORE_READY = 10

#: Appended to every rung. Judges reliably repair a vague hypothesis out of
#: their own domain knowledge -- supplying the obvious metric, the conventional
#: baseline -- which is precisely the failure this metric exists to detect.
_NO_REPAIR = (
    " Judge only what the research goal and the hypothesis actually say. Do not supply the "
    "missing piece from your own domain knowledge, and do not credit the hypothesis for "
    "something a reader would conventionally assume. If you find yourself inferring it, the "
    "answer is no."
)

INTERVENTION_CRITERIA = (
    "Does the hypothesis name a specific intervention, treatment, or independent variable that an "
    "experimenter would deliberately manipulate, vary, or apply? A named research topic, "
    "application area, or system to be studied is not an intervention; the text must identify "
    "what would actually be changed between conditions." + _NO_REPAIR
)

MEASUREMENT_CRITERIA = (
    "Does the hypothesis name a specific outcome, dependent variable, or observable that would be "
    "recorded, concretely enough to know what instrument or log would produce it? A named metric, "
    "quantity, rate, or measurable property counts. An unquantified promise such as 'better "
    "performance', 'improved efficiency', or 'higher quality' does not." + _NO_REPAIR
)

COMPARISON_CRITERIA = (
    "Does the hypothesis state what the measured outcome would be compared against -- a control "
    "condition, an untreated group, a named baseline method, an alternative configuration, or an "
    "explicit prior value or target? The comparator must be present in the text; a bare claim of "
    "improvement with nothing to improve upon does not count." + _NO_REPAIR
)

PREDICTION_CRITERIA = (
    "Does the hypothesis predict the direction or size of the effect -- that the outcome will "
    "increase, decrease, or outperform the comparator, or that it will reach a stated magnitude or "
    "threshold? Saying the effect will be investigated, explored, assessed, or evaluated is not a "
    "prediction, because no result could contradict it." + _NO_REPAIR
)


def build_experimental_readiness_dag() -> DeepAcyclicGraph:
    """Return a fresh readiness ladder.

    Built per metric rather than shared: nodes cache their verdict during
    ``measure``, so one graph reused across hypotheses would report the first
    hypothesis's answers for all of them.
    """
    intervention = BinaryJudgementNode(
        criteria=INTERVENTION_CRITERIA,
        evaluation_params=EVALUATION_PARAMS,
        label="intervention",
    )
    measurement = BinaryJudgementNode(
        criteria=MEASUREMENT_CRITERIA,
        evaluation_params=EVALUATION_PARAMS,
        label="measurement",
    )
    comparison = BinaryJudgementNode(
        criteria=COMPARISON_CRITERIA,
        evaluation_params=EVALUATION_PARAMS,
        label="comparison",
    )
    prediction = BinaryJudgementNode(
        criteria=PREDICTION_CRITERIA,
        evaluation_params=EVALUATION_PARAMS,
        label="prediction",
    )

    # Top-down construction: passing `children=` to a node is the deprecated
    # bottom-up API in DeepEval 4.2.2.
    intervention.add_verdict(False, score=SCORE_NO_INTERVENTION)
    intervention.add_verdict(True, then=measurement)

    measurement.add_verdict(False, score=SCORE_NO_MEASUREMENT)
    measurement.add_verdict(True, then=comparison)

    comparison.add_verdict(False, score=SCORE_NO_COMPARISON)
    comparison.add_verdict(True, then=prediction)

    prediction.add_verdict(False, score=SCORE_NO_PREDICTION)
    prediction.add_verdict(True, score=SCORE_READY)

    return DeepAcyclicGraph(root_nodes=[intervention])


__all__ = [
    "COMPARISON_CRITERIA",
    "EVALUATION_PARAMS",
    "INTERVENTION_CRITERIA",
    "MEASUREMENT_CRITERIA",
    "PREDICTION_CRITERIA",
    "SCORE_NO_COMPARISON",
    "SCORE_NO_INTERVENTION",
    "SCORE_NO_MEASUREMENT",
    "SCORE_NO_PREDICTION",
    "SCORE_READY",
    "build_experimental_readiness_dag",
]
