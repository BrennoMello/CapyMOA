import moa.classifiers.core.driftdetection as drift_detection
import moa.classifiers.trees as moa_trees
from capymoa.base import Classifier
from moa.core import Utils
from jpype import _jpype
import numpy as np
import math
import copy

class ConceptDriftMethodClassifier(Classifier):

    def __init__(
        self, 
        schema=None, 
        random_seed=1, 
        moa_drift_detector=None, 
        moa_learner=None,
        loss=None,
    ):
        self.DDM_INCONTROL_LEVEL = 0
        self.DDM_WARNING_LEVEL = 1
        self.DDM_OUTCONTROL_LEVEL = 2

        self.warning_detected = 0
        self.change_detected = 0

        #TODO: change the class start
        # self.moa_learner_option = moa_trees.HoeffdingTree
        # self.moa_drift_detection_option = drift_detection.ADWINChangeDetector

        # moa_drift_detector = self.moa_drift_detection_option
        # moa_learner = self.moa_learner_option

        self.loss_function = loss() if loss is not None else None

        self.random_seed = random_seed
        self.schema = schema

        self.moa_learner = moa_learner
        self.moa_drift_detector = moa_drift_detector

        self.classifier = moa_learner
        self.new_classifier = copy.deepcopy(self.moa_learner)
        
        self.drift_detection_method = self.moa_drift_detector

        # self.classifier.setRandomSeed(self.random_seed)
        
        # if self.schema is not None:
        #     self.classifier.setModelContext(self.schema.get_moa_header())

        # self.classifier.prepareForUse()
        # self.classifier.resetLearningImpl()
        # self.classifier.setModelContext(schema.get_moa_header())

    def train(self, instance, delay=0):
        # instance = instance.java_instance
        true_class = instance.java_instance.getData().classValue()

        predict_class = self.classifier.predict(instance)
        predict_proba = self.classifier.predict_proba(instance)
        print(f"predict_proba: {predict_proba}")
        if predict_class == true_class:
            prediction = True
        else:
            prediction = False
        
        #TODO: Change the loss function 
        # self.drift_detection_method.input(0.0 if prediction else 1.0)
        
        loss_value = self.loss_function.input(true_class, predict_proba, delay)
        self.drift_detection_method.add_element(loss_value)
        self.ddmLevel = self.DDM_INCONTROL_LEVEL

        if self.drift_detection_method.detected_change():
            self.ddmLevel = self.DDM_OUTCONTROL_LEVEL
        
        if self.drift_detection_method.detected_warning():
            self.ddmLevel =  self.DDM_WARNING_LEVEL

        if self.ddmLevel == self.DDM_WARNING_LEVEL:
            print("DDM_WARNING_LEVEL")
            if self.new_classifier_reset == True:
                self.warning_detected += 1
                self.new_classifier.reset()
                self.new_classifier_reset = False
            
            # self.new_classifier.trainOnInstance(instance)
            self.new_classifier.train(instance)
            

        elif self.ddmLevel == self.DDM_OUTCONTROL_LEVEL:
            print("DDM_OUTCONTROL_LEVEL")
            self.change_detected += 1
            self.classifier = self.new_classifier
            # self.classifier = self.new_classifier.copy()

            # if isinstance(self.classifier, WEKAClassifier):
            #     self.classifier.buildClassifier()
            # self.new_classifier = self.moa_learner_option().copy()
            self.new_classifier = copy.deepcopy(self.moa_learner)
            self.new_classifier.reset()
            

        elif self.ddmLevel == self.DDM_INCONTROL_LEVEL:
            # print("DDM_INCONTROL_LEVEL")
            self.new_classifier_reset = True
            
        self.classifier.train(instance)

    def __str__(self):
        return str("ConceptDriftMethodClassifier")

    def predict(self, instance):
        # return Utils.maxIndex(
        #     self.classifier.getVotesForInstance(instance.java_instance)
        # )
        return self.classifier.predict(instance)

    def predict_proba(self, instance):
        # return self.classifier.getVotesForInstance(instance.java_instance)
        return self.classifier.predict_proba(instance)
    
    # def get_model_measurements(self):
    #     measurement_list = list()
    #     measurement_list.append(("Change detected", self.change_detected))
    #     measurement_list.append(("Warning detected", self.warning_detected))
    #     model_measurements = self.classifier.getModelMeasurements()
    #     if model_measurements != None:
    #         for measurement in model_measurements:
    #             measurement_list.append((measurement.getName(), measurement.getValue()))
            
    #     self.change_detected = 0
    #     self.warning_detected = 0
        
    #     return measurement_list

    #TODO: Update the get output function
    def get_cd_votes(self, instance):
        return self.drift_detection_method.get_votes()

    def resetLearning(self):
        # self.classifier = copy.deepcopy(self.moa_learner)
        self.classifier = self.moa_learner.copy()
        self.new_classifier = self.moa_learner.copy()
        self.classifier.reset()
        self.new_classifier.reset()
        # self.driftDetectionMethod = self.moa_drift_detection.copy()
        self.driftDetectionMethod.reset(clean_history=True)
        self.new_classifier_reset = False
        


class LSD():
    
    def input(self, true_class, predict_proba, delay):
       return self._penalize_loss(true_label=true_class, predicted_probs=predict_proba, delay=delay)

    def _categorical_cross_entropy(self, y_true, y_pred):
        # Avoid log(0) by adding a small epsilon value
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
        # Compute the loss (sum over all classes for each instance)
        loss = -np.sum(y_true * np.log10(y_pred))
        return loss

    def _penalize_loss(self, true_label, predicted_probs, delay, k=0.1):
        loss = self._categorical_cross_entropy(true_label, predicted_probs)
        return loss + (1 - loss) * (1 - math.exp(-k * delay))