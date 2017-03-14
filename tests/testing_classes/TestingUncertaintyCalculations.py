from uncertainpy import UncertaintyCalculations

class TestingUncertaintyCalculations(UncertaintyCalculations):
    def PC(self, uncertain_parameters=None, method="regression", rosenblatt=False, plot_condensed=True):
        arguments = {}

        arguments["function"] = "PC"
        arguments["uncertain_parameters"] = uncertain_parameters
        arguments["method"] = method
        arguments["rosenblatt"] = rosenblatt
        arguments["plot_condensed"] = plot_condensed

        return arguments


    def MC(self, uncertain_parameters=None, plot_condensed=True):
        arguments = {}

        arguments["function"] = "MC"
        arguments["uncertain_parameters"] = uncertain_parameters
        arguments["plot_condensed"] = plot_condensed


        return arguments



    def CustomUQ(self, custom_keyword="custom_value"):
        arguments = {}

        arguments["function"] = "CustomUQ"
        arguments["custom_keyword"] = custom_keyword

        return arguments