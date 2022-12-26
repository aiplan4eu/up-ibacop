import os
import unified_planning as up
from unified_planning.engines.mixins import PortfolioSelectorMixin
from unified_planning.engines import Engine, Credits
from unified_planning.io.pddl_writer import PDDLWriter
from unified_planning.exceptions import UPException
from typing import List
from utils.models import joinFile
import tempfile

credits = Credits('IBACOP',
                  'Isabel(?)',
                  '*@gmail.com',
                  'https://*/',
                  '?',
                  '?',
                  '?')


class Ibacop(PortfolioSelectorMixin, Engine):
    def __init__(self, model_path, dataset_path):
        super().__init__(model_path)
        planner_list = self._init_planner_list(dataset_path)

    def _init_planner_list(self, dataset_path):
        word = "@attribute planner"
        with open(dataset_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.find(word) != -1:
                    line = line.replace(word, "")
                    line = line.replace("{", "")
                    line = line.replace("}", "")
                    line = line.replace(" ", "")
                    self.planner_list = line.strip().split(",")

    def extract_features(
        self,
        problem: "up.model.AbstractProblem" = None,
        domain_path: str = None,
        problem_path: str = None,
    ) -> List[str]:

        current_path = os.path.dirname(__file__)
        current_wdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tempdir:
            if problem is not None:
                w = PDDLWriter(problem, True)
                domain_filename = os.path.join(tempdir, "domain.pddl")
                problem_filename = os.path.join(tempdir, "problem.pddl")
                w.write_domain(domain_filename)
                w.write_problem(problem_filename)
            elif domain_path is not None and problem_path is not None:
                domain_filename = domain_path
                problem_filename = problem_path
            else:
                raise UPException(
                    "You need to pass an AbstractProblem or the domain and problem path"
                )

            # need to change the working dir for the following commands to work properly
            os.chdir(tempdir)
            print("\n***start extract features***\n")
            # features
            command = (
                "python2.7 "
                + current_path
                + "/features/translate/translate.py "
                + domain_filename
                + " "
                + problem_filename
            )
            print(command)
            os.system(command)

            command = (
                current_path
                + "/features/preprocess/preprocess < "
                + tempdir
                + "/output.sas"
            )
            os.system(command)

            command = (
                current_path
                + "/features/ff-learner/roller3.0 -o "
                + domain_filename
                + " -f "
                + problem_filename
                + " -S 28"
            )
            os.system(command)

            command = (
                current_path
                + "/features/heuristics/training.sh "
                + domain_filename
                + " "
                + problem_filename
            )
            os.system(command)

            command = (
                current_path
                + '/search/downward --landmarks "lm=lm_merged([lm_hm(m=1),lm_rhw(),lm_zg()])" < '
                + tempdir
                + "/output"
            )
            os.system(command)

            command = (
                current_path
                + "/search-mercury/downward ipc seq-agl-mercury <"
                + tempdir
                + "/output"
            )
            os.system(command)

            # join file
            temp_result = []
            for p in self.planner_list:
                temp_result.append(p + ",?")

            joinFile.create_globals(tempdir, temp_result, self.planner_list)

            print("\n***end extract features***\n")

            # go back to the previously working dir
            os.chdir(current_wdir)
            with open(os.path.join(tempdir, "global_features.arff")) as file_features:
                return file_features.readlines()

    def create_model(self, features) -> str:
        # per funzionare ha bisogno di:
        # weka all'interno della dir current_path/models/
        # e crea il .model in current_path (ovvero dove c'è il file ibacop.py)
        current_path = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tempdir:
            features_path = os.path.join(tempdir, "global_features.arff")

            with open(features_path, "w") as file:
                for line in features:
                    # write each item on a new line
                    file.write("%s\n" % line)

            # Call to `weka.jar` to remove unused `features`
            command = (
                "java -cp "
                + current_path
                + "/models/weka.jar -Xms256m -Xmx1024m weka.filters.unsupervised.attribute.Remove -R 1-3,18,20,65,78-79,119-120 -i "
                + features_path
                + " -o "
                + tempdir
                + "/global_features_simply.arff"
            )
            os.system(command)

            # Save the model created
            command = (
                "java -Xms256m -Xmx1024m -cp "
                + current_path
                + "/models/weka.jar weka.classifiers.meta.RotationForest  -t "
                + tempdir
                + "/global_features_simply.arff -d "
                + current_path
                + "/RotationForest.model"
            )
            os.system(command)

            model_path = os.path.join(current_path, "RotationForest.model")
            if os.path.isfile(model_path):
                return model_path
            else:
                return None

    def get_prediction(self, features) -> List[str]:
        current_path = os.path.dirname(__file__)
        current_wdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tempdir:

            features_path = os.path.join(tempdir, "global_features.arff")
            with open(features_path, "w") as file:
                for line in features:
                    # write each item on a new line
                    file.write("%s\n" % line)

            os.chdir(tempdir)

            # Call to `weka.jar` to remove unused `features`
            command = (
                "java -cp "
                + current_path
                + "/models/weka.jar -Xms256m -Xmx1024m weka.filters.unsupervised.attribute.Remove -R 1-3,18,20,65,78-79,119-120 -i "
                + features_path
                + " -o "
                + tempdir
                + "/global_features_simply.arff"
            )
            os.system(command)
            # far predirre a weka
            # ottiene la lista dei pianificatori ordinata per probabilità di successo
            command = (
                "java -Xms256m -Xmx1024m -cp "
                + current_path
                + "/models/weka.jar weka.classifiers.meta.RotationForest -l "
                + self.model_path
                + " -T "
                + tempdir
                + "/global_features_simply.arff -p 113 > "
                + tempdir
                + "/outputModel"
            )
            os.system(command)
            # The `model` creates the `list` of ALL planners relative to their probability of solving the `problem`
            command = (
                "python2.7 "
                + current_path
                + "/models/parseWekaOutputFile.py "
                + tempdir
                + "/outputModel "
                + tempdir
                + "/listPlanner"
            )
            os.system(command)

            # go back to the previously working dir
            os.chdir(current_wdir)
            with open(os.path.join(tempdir, "listPlanner"), "r") as file:
                return file.readlines()