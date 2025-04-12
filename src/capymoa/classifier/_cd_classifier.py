from capymoa.base import MOAClassifier
from moa.core import Utils
from jpype import _jpype

class ConceptDriftMethodClassifier(MOAClassifier):

    def __init__(
        self, 
        schema=None, 
        random_seed=1, 
        moa_drift_detector=None, 
        moa_learner=None
    ):
        self.DDM_INCONTROL_LEVEL = 0
        self.DDM_WARNING_LEVEL = 1
        self.DDM_OUTCONTROL_LEVEL = 2

        self.warning_detected = 0
        self.change_detected = 0

        self.moa_drift_detectot_option = moa_drift_detector
        self.moa_learner_option = moa_learner
        
        # If moa_learner is a class identifier instead of an object
        if isinstance(moa_learner, type):
            if isinstance(moa_learner, _jpype._JClass):
                moa_learner = moa_learner()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise ValueError("Invalid MOA classifier provided.")

        if isinstance(moa_drift_detector, type):
            if isinstance(moa_drift_detector, _jpype._JClass):
                moa_drift_detector = moa_drift_detector()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise ValueError("Invalid MOA drift detector provided.")   

        self.classifier = moa_learner
        self.classifier.setRandomSeed(self.random_seed)
        self.new_classifier = moa_learner

        self.drift_detection_method = moa_drift_detector

    def train(self, instance):
        instance = instance.java_instance
        true_class = instance.value()

        predic_class = Utils.maxIndex(
                            self.classifier.getVotesForInstance(instance)
                        )
        
        if predic_class == true_class:
            prediction = True
        else:
            prediction = False
        
        self.drift_detection_method(0.0 if prediction else 1.0)
        self.ddmLevel = self.DDM_INCONTROL_LEVEL

        if self.drift_detection_method.getChange():
            self.ddmLevel = self.DDM_OUTCONTROL_LEVEL
        
        if self.drift_detection_method.getWarningZone():
            self.ddmLevel =  self.DDM_WARNING_LEVEL

        if self.ddmLevel == self.DDM_WARNING_LEVEL:
            print("DDM_WARNING_LEVEL")
            if self.new_classifier_reset is True:
                self.warning_detected += 1
                self.new_classifier.resetLearning()
                self.new_classifier_reset = False
            
            self.new_classifier.trainOnInstance(instance)

        elif self.ddmLevel == self.DDM_OUTCONTROL_LEVEL:
            print("DDM_OUTCONTROL_LEVEL")
            self.change_detected += 1
            self.classifier = self.new_classifier
            # if isinstance(self.classifier, WEKAClassifier):
            #     self.classifier.buildClassifier()
            self.new_classifier = self.moa_learner_option().copy()
            self.new_classifier.resetLearning()

        elif self.ddmLevel == self.DDM_INCONTROL_LEVEL:
            print("DDM_INCONTROL_LEVEL")
            self.new_classifier_reset = True
            return

        self.classifier.trainOnInstace(instance)

    def get_model_measurements(self):
        measurement_list = list()
        measurement_list.add(("Change detected", self.change_detected))
        measurement_list.add(("Warning detected", self.warning_detected))
        model_measurements = self.classifier.getModelMeasurements()
        if model_measurements != None:
            for measurement in model_measurements:
                measurement_list.add((measurement.getName(), measurement.getValue()))
            
        self.change_detected = 0;
        self.warning_detected = 0;
        
        return measurement_list
        

    def get_votes_for_instance(self, instance):
        return self.drift_detection_method.getOutput()
        

    def resetLearning(self):
        self.classifier = self.moa_learner_option().copy()
        self.new_classifier = self.classifier.copy()
        self.classifier.resetLearning()
        self.new_classifier.resetLearning()
        self.driftDetectionMethod =  self.moa_drift_detectot_option().copy()
        self.new_classifier_reset = False
        
        