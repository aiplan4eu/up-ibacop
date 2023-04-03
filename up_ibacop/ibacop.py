import os
import unified_planning as up
from unified_planning.shortcuts import get_env
from unified_planning.model import ProblemKind
from unified_planning.engines.mixins import PortfolioSelectorMixin
from unified_planning.engines import Engine, Credits
from unified_planning.io.pddl_writer import PDDLWriter
from unified_planning.exceptions import UPUsageError
from typing import Any, Dict, List, Optional, Tuple
from up_ibacop.utils.models import joinFile
from up_ibacop.utils.models import parseWekaOutputFile
import tempfile
import ast
import subprocess

credits = Credits(
    "IBaCoP2",
    "Isabel Cenamor and Tomas de la Rosa and Fernando Fernandez",
    "icenamorg@gmail.com",
    " ",
    "GPL",
    "Instance Based Configured Portfolios ",
    "IBaCoP2 is a system for the configuration of a portfolio of planners based on the features of a problem instance",
)


rootpath = os.path.dirname(__file__)
default_model_path = os.path.join(rootpath, "model", "RotationForest.model")
default_dataset_path = os.path.join(rootpath, "model", "global_features_simply.arff")


def extract_tuple_from_list(
    tuple_list: List[str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """This method takes a list of tuples in string format and returns them with the right format"""
    planners = []
    parameters = []
    for tuple in tuple_list:
        tmp = tuple.split("|")
        planner_name = tmp[0]
        planner_parameters = tmp[1]
        # Can't save the parameters list with {} because they represent a special character for weka
        planner_parameters = planner_parameters.replace(";", ",")
        planner_parameters = "{" + planner_parameters + "}"
        planner_parameters_dict = ast.literal_eval(planner_parameters)

        planners.append(planner_name)
        parameters.append(planner_parameters_dict)

    return planners, parameters


def init_planners_data() -> Tuple[List[str], List[Dict[str, Any]]]:
    word = "@attribute planner"
    with open(default_dataset_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.find(word) != -1:
                line = line.replace(word, "")
                line = line.replace("{", "")
                line = line.replace("}", "")
                line = line.replace(" ", "")
                model_planners, model_parameters = extract_tuple_from_list(
                    line.strip().split(",")
                )

        return model_planners, model_parameters


default_planners, default_parameters = init_planners_data()


class Ibacop(PortfolioSelectorMixin, Engine):
    def __init__(self):
        Engine.__init__(self)
        PortfolioSelectorMixin.__init__(self)

    @property
    def name(self) -> str:
        return "ibacop"

    @staticmethod
    def supported_kind() -> ProblemKind:
        raise UPUsageError

    @staticmethod
    def supports(problem_kind: "ProblemKind") -> bool:
        factory = get_env().factory
        installed_planners = factory.engines

        for planner in default_planners:
            if planner in installed_planners:
                if factory.engine(planner).supports(problem_kind):
                    return True
        return False

    @staticmethod
    def get_credits(**kwargs) -> Optional[Credits]:
        return credits

    @staticmethod
    def satisfies(
        optimality_guarantee: "up.engines.mixins.oneshot_planner.OptimalityGuarantee",
    ) -> bool:
        factory = get_env().factory
        installed_planners = factory.engines

        for planner in default_planners:
            if planner in installed_planners:
                if not factory.engine(planner).satisfies(optimality_guarantee):
                    return False
        return True

    def _get_best_oneshot_planners(
        self,
        problem: "up.model.AbstractProblem",
        max_planners: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:

        features = self._extract_features(problem)
        model_prediction_list = self._get_prediction(features)

        model_prediction_list = self._filter_with_system_planners(model_prediction_list)

        n_selected_planners = 0
        list_planners = []
        for planner in model_prediction_list:
            planner = planner.strip()
            list_planners.append(planner)
            n_selected_planners += 1
            if n_selected_planners == max_planners:
                break

        return extract_tuple_from_list(list_planners)

    def _extract_features(self, problem: "up.model.AbstractProblem") -> List[str]:
        """This method extracts the features of the 'problem' in input and returns them as a List[str]"""
        current_path = os.path.dirname(__file__)
        current_wdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tempdir:
            w = PDDLWriter(problem, True)
            domain_filename = os.path.join(tempdir, "domain.pddl")
            problem_filename = os.path.join(tempdir, "problem.pddl")
            w.write_domain(domain_filename)
            w.write_problem(problem_filename)

            # Need to change the working dir for the following commands to work properly
            os.chdir(tempdir)

            translate_path = os.path.join(
                current_path, "utils", "features", "translate", "translate.py"
            )
            command = (
                "python "
                + translate_path
                + " "
                + domain_filename
                + " "
                + problem_filename
                + " 2> /dev/null"
            )
            p = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.communicate()

            preprocess_path = os.path.join(
                current_path, "utils", "features", "preprocess", "preprocess"
            )
            output_sas_path = os.path.join(tempdir, "output.sas")
            if os.path.isfile(output_sas_path):
                command = preprocess_path + " < " + output_sas_path + " 2> /dev/null"
                p = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                p.communicate()

            roller_path = os.path.join(
                current_path, "utils", "features", "ff-learner", "roller3.0"
            )
            command = (
                roller_path
                + " -o "
                + domain_filename
                + " -f "
                + problem_filename
                + " -S 28"
                + " 2> /dev/null"
            )
            p = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.communicate()

            training_sh_path = os.path.join(
                current_path, "utils", "features", "heuristics", "training.sh"
            )
            command = (
                training_sh_path
                + " "
                + domain_filename
                + " "
                + problem_filename
                + " 2> /dev/null"
            )
            p = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.communicate()

            downward_path = os.path.join(current_path, "utils", "search", "downward")
            output_path = os.path.join(tempdir, "output")
            if os.path.isfile(output_path):
                command = (
                    downward_path
                    + ' --landmarks "lm=lm_merged([lm_hm(m=1),lm_rhw(),lm_zg()])" < '
                    + output_path
                    + " 2> /dev/null"
                )
                p = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                p.communicate()

            mercury_downward_path = os.path.join(
                current_path, "utils", "search-mercury", "downward"
            )
            if os.path.isfile(output_path):
                command = (
                    mercury_downward_path
                    + " ipc seq-agl-mercury <"
                    + output_path
                    + " 2> /dev/null"
                )
                p = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                p.communicate()

            # Formatting the list of names and the list of parameters into a list of tuples to be used by weka
            tuple_list = []
            for i in range(0, len(default_planners)):
                tmp_str = default_planners[i] + "|" + str(default_parameters[i])
                tmp_str = tmp_str.replace("{", "")
                tmp_str = tmp_str.replace("}", "")
                tmp_str = tmp_str.replace(",", ";")
                tuple_list.append(tmp_str)

            temp_result = []
            for t in tuple_list:
                temp_result.append(str(t) + ",?")

            joinFile.create_globals(tempdir, temp_result, tuple_list)

            # Return to the previous working dir
            os.chdir(current_wdir)

            with open(os.path.join(tempdir, "global_features.arff")) as file_features:
                return file_features.readlines()

    def _get_prediction(self, features: List[str]) -> List[str]:
        """This method takes the features and returns a sorted list of planners created by weka using a trained model"""
        current_path = os.path.dirname(__file__)
        current_wdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tempdir:

            features_path = os.path.join(tempdir, "global_features.arff")
            with open(features_path, "w") as file:
                for line in features:
                    file.write("%s\n" % line)

            # Need to change the working dir for the following commands to work properly
            os.chdir(tempdir)

            # Call to 'weka.jar' to remove unused 'features'
            command = (
                "java -cp "
                + current_path
                + "/utils/models/weka.jar -Xms256m -Xmx1024m weka.filters.unsupervised.attribute.Remove -R 1-3,18,20,65,78-79,119-120 -i "
                + features_path
                + " -o "
                + tempdir
                + "/global_features_simply.arff"
            )
            # os.system(command)
            p = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.communicate()

            # Weka returns the predictions
            command = (
                "java -Xms256m -Xmx1024m -cp "
                + current_path
                + "/utils/models/weka.jar weka.classifiers.meta.RotationForest -l "
                + default_model_path
                + " -T "
                + tempdir
                + "/global_features_simply.arff -p 113 > "
                + tempdir
                + "/outputModel"
            )
            p = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.communicate()

            parseWekaOutputFile.parseOutputFile(
                os.path.join(tempdir, "outputModel"),
                os.path.join(tempdir, "listPlanner"),
            )

            # Return to the previous working dir
            os.chdir(current_wdir)

            with open(os.path.join(tempdir, "listPlanner"), "r") as file:
                return file.readlines()

    def _filter_with_system_planners(
        self, planner_list: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        installed_planners = get_env().factory.engines

        planners = []
        for planner in planner_list:
            planner = planner.strip()
            delimiter = planner.find("|")
            if planner[0:delimiter] in installed_planners:
                planners.append(planner)

        return planners
