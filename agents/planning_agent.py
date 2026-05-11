from agent import Agent

class PlanningAgent(Agent):
    def __init__(self):
        super().__init__("PlanningAgent")
        
    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message
        """
        mapping = {
            # "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            # "estimate_true_value": self.estimate_true_value,
            # "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results