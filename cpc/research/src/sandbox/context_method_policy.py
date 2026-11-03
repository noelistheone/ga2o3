"""Operational aliases for the independently saved context-property predictor."""
from pathlib import Path
import json
from .context_property_predictor import ContextPropertyPredictor


class ValidatedContextPolicy:
    def __init__(self,bundle,policies):
        self.base=ContextPropertyPredictor(bundle)
        self.policy=json.loads(Path(policies).read_text())
        if self.policy['version']!=1:raise ValueError('Unknown saved method-policy version')
        for choices in self.policy['aliases'].values():
            for preset in choices.values():
                if preset not in self.base.manifest['presets']:raise ValueError('Policy points to an absent saved model')

    def predict(self,*,policy,inputs,anchors=()):
        if policy not in self.policy['aliases']:raise ValueError('Unknown saved policy')
        if len(anchors)>2:raise ValueError('Saved method policies support0/1/2Hall anchors')
        preset=self.policy['aliases'][policy][str(len(anchors))]
        out=self.base.predict(preset=preset,inputs=inputs,anchors=anchors)
        out.update(selected_policy=policy,selected_variant=self.base.manifest['presets'][preset]['variant'],
            selection_scope='model and source penalties selected on training-inner labels;newsource query labels unused')
        return out
