"""Declared-protocol consistency, not measured transfer or FM certification.

Database IDs must be canonical provenance identities (including aliases/copies).
The caller supplies reviewed metadata; these functions cannot authenticate it.
"""
def adaptation_route(backbone_updates, head_updates, target_labels):
    """Describe a simplified adaptation route, with weight updates taking priority.

Counts cover target-task adaptation, not original pretraining. Target labels
include all target-task examples supplied to the learner, not test outcomes.
Hybrid prompting plus fine-tuning is reported under FINE_TUNING.
"""
    counts = (backbone_updates, head_updates, target_labels)
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('Update and label counts must be nonnegative integers')
    if backbone_updates:
        return 'FINE_TUNING'
    if head_updates:
        return 'FROZEN_ENCODER_HEAD'
    return 'IN_CONTEXT' if target_labels else 'ZERO_SHOT'


def database_boundary(pretraining_databases, evaluation_database):
    """Check pretraining membership; target adaptation is deliberately separate."""
    if not isinstance(evaluation_database, str) or not evaluation_database or evaluation_database.strip() != evaluation_database:
        raise ValueError('Use a nonempty canonical evaluation database ID')
    if pretraining_databases is None:
        return 'UNKNOWN'
    if not isinstance(pretraining_databases, list):
        raise ValueError('Pretraining inventory must be a list or None')
    if any(not isinstance(x, str) or not x or x.strip() != x for x in pretraining_databases):
        raise ValueError('Use nonempty canonical pretraining database IDs')
    if len(set(pretraining_databases)) != len(pretraining_databases):
        raise ValueError('Duplicate IDs: resolve database lineage before counting')
    if not pretraining_databases:
        return 'UNKNOWN'
    return 'SEEN' if evaluation_database in pretraining_databases else 'HELD_OUT'


def scope_verdict(record, route=adaptation_route, boundary=database_boundary):
    """Audit a declared held-out-database comparison; never certify a model.

READY_FOR_REVIEW means the supplied metadata is internally consistent. It
requires independent source, prediction and provenance review before any claim.
"""
    required = {'pretraining_databases','evaluation_database','backbone_updates',
                'head_updates','target_labels','declared_mode','selection_split',
                'evaluation_split','test_labels_used','temporal_audit',
                'baseline_matched','completed','checkpoint_shared'}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError('Missing scope fields')
    for key in ['test_labels_used','baseline_matched','completed','checkpoint_shared']:
        if type(record[key]) is not bool:
            raise ValueError(key + ' must be a Boolean')
    if record['temporal_audit'] not in ['PASS','FAIL','NOT_ESTABLISHED']:
        raise ValueError('Unknown temporal audit status')
    if record['selection_split'] not in ['train','validation','test'] or record['evaluation_split'] not in ['train','validation','test']:
        raise ValueError('Unknown split')
    if record['declared_mode'] not in ['ZERO_SHOT','IN_CONTEXT','FROZEN_ENCODER_HEAD','FINE_TUNING']:
        raise ValueError('Unknown adaptation declaration')
    mode = route(record['backbone_updates'], record['head_updates'], record['target_labels'])
    membership = boundary(record['pretraining_databases'], record['evaluation_database'])
    issues = []
    if membership == 'UNKNOWN': issues.append('PRETRAINING_UNKNOWN')
    if membership == 'SEEN': issues.append('PRETRAINING_OVERLAP')
    if record['selection_split'] == 'test': issues.append('TEST_SELECTION')
    if record['evaluation_split'] != 'test': issues.append('NO_HELD_OUT_TEST')
    if record['test_labels_used']: issues.append('TEST_LABEL_ACCESS')
    if record['temporal_audit'] != 'PASS': issues.append('TEMPORAL_UNVERIFIED')
    if not record['baseline_matched']: issues.append('BASELINE_UNMATCHED')
    if not record['checkpoint_shared']: issues.append('NO_SHARED_CHECKPOINT')
    if mode != record['declared_mode']: issues.append('MODE_MISMATCH')
    status = 'REVISE' if issues else 'READY_FOR_REVIEW' if record['completed'] else 'READY_TO_RUN'
    return dict(mode=mode, pretraining_boundary=membership, status=status, issues=issues,
                universal_claim='NOT_ESTABLISHED', learner='PENDING_WRITTEN_DEFENSE')
