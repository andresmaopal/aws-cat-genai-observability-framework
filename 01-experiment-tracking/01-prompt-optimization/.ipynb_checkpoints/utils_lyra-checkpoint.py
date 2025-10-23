import json
import datetime
import ipywidgets as widgets
from IPython.display import display, HTML
from strands import Agent
from strands.models import BedrockModel
import sys
from io import StringIO

LYRA_PROMPT = """You are Lyra, an AI prompt optimization specialist. Transform user inputs into token-efficient, high-impact prompts using the 4-D methodology.

<core_principle>
Find the minimal set of high-signal tokens that maximize desired outcomes without dimishing instructions clarity and information. Every token must justify its presence.
</core_principle>

<methodology>
1. DECONSTRUCT: Extract core intent, map requirements vs. optional elements
2. DIAGNOSE: Audit for clarity gaps, bloated instructions, missing critical context
3. DEVELOP: Apply context-efficient techniques by request type:
- Creative → Minimal role + clear constraints
- Technical → Structured sections + precision focus
- Educational → Curated examples only
- Complex → Progressive disclosure + frameworks
4. DELIVER: Token-efficient prompts with clear structure
</methodology>

<operating_modes>
ADVANCED: Gather essential missing context (max 5 questions), then optimize
EFFICIENT: Immediate optimization addressing core gaps
</operating_modes>

<critical_context_check>
Assess if missing: Audience, Success criteria, Format constraints, Domain context, Scope boundaries
</critical_context_check>

<output_format>
Always respond with this exact format without using Markdown, only plain text:

1. If there are questions to the user (ADVANCED mode):

QUESTIONS TO ENHANCE THE PROMPT:
--------------------------------------------------------------------------------------------------

Let me ask a few key questions:

1. TOPIC OF QUESTION: Question 1 here
2. TOPIC OF QUESTION: Question 2 here

2. If there are NO questions to the user (EFFICIENT mode):

OPTIMIZED PROMPT -----------------------------------------------------------------------------------

```
[The optimized prompt with XML tags like <role>, <task>, <context>, <constraints>, etc.]
```

END OF PROMPT ---------------------------------------------------------------------------------------

KEY OPTIMIZATIONA MADE:

• [Bullet point 1 describing what was optimized, dont use markdown]
• [Bullet point 2 describing what was optimized, dont use markdown]
• [Additional bullet points as needed]

IMPORTANT: Include XML tags in the code block when structuring the prompt (e.g., <role>, <task>, <context>, <constraints>, <output_format>, etc.). These tags are essential for prompt structure.

If you need clarification, ask your questions first, then provide the optimized prompt with proper XML tags and optimization bullets.
</output_format>

<welcome_protocol>
Request user's mode preference (STRATEGIC/EFFICIENT) and target (Claude/General), then process their prompt using 4-D methodology.
</welcome_protocol>

<standards>
- Use XML tags for structure when appropriate
- Parentheses () not brackets []
- Eliminate redundant content
- Focus on behavior-driving information
- Output ONLY the final optimized prompt
</standards>"""

class LyraOptimizer:
    def __init__(self, model_name="claude-4-sonnet"):
        """Initialize Lyra with specified model"""
        with open('model_list.json', 'r') as f:
            models = json.load(f)
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found in model_list.json")
        
        model_config = models[model_name]
        self.model = BedrockModel(
            model_id=model_config["model_id"],
            region_name=model_config["region_name"],
            temperature=model_config.get("temperature", 0.1)
        )
        
        self.agent = Agent(
            model=self.model,
            system_prompt=LYRA_PROMPT
        )
    
    def optimize_prompt(self, user_input):
        """Process user input through Lyra optimization"""
        try:
            response = self.agent(user_input)
            # Extract text from AgentResult object
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_welcome_message(self):
        """Get Lyra's welcome message"""
        welcome_msg = """Welcome! I'm Lyra, your prompt optimization specialist.

1. Please choose your preferred mode:

- ADVANCED: I'll ask up to 5 questions to gather essential context, then optimize.
- EFFICIENT: I'll immediately optimize your prompt addressing core gaps.

2. Also specify your target model:

- Claude: Optimized for Anthropic's Claude models.
- General: Optimized for general LLM use.

3. PASTE the prompt you'd like me to optimize!"""
        return welcome_msg

class LyraChat:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.chat_history = []
        
        # Create widgets
        self.output = widgets.Output()
        self.input_text = widgets.Textarea(
            placeholder="Enter your prompt to optimize...",
            layout=widgets.Layout(width='100%', height='300px')
        )
        self.send_button = widgets.Button(
            description="Send",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.clear_button = widgets.Button(
            description="Clear Chat",
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        
        # Bind events
        self.send_button.on_click(self.send_message)
        self.clear_button.on_click(self.clear_chat)
    
    def display_xml_message(self, sender, message, msg_type):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if msg_type == "user":
            style = "background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin: 5px 0;"
            icon = "👤"
        else:
            style = "background-color: #f3e5f5; border-left: 4px solid #9c27b0; padding: 10px; margin: 5px 0;"
            icon = "🎯"
        
        # Escape HTML but preserve XML tags for display
        import html
        escaped_message = html.escape(message)
        
        html_content = f"""
        <div style="{style}">
            <strong>{icon} {sender}</strong> <small style="color: #666;">({timestamp})</small><br>
            <div style="margin-top: 5px; white-space: pre-wrap; font-family: monospace; background-color: #f8f9fa; padding: 8px; border-radius: 4px; border: 1px solid #e9ecef;">{escaped_message}</div>
        </div>
        """
        display(HTML(html_content))
    
    def display_message(self, sender, message, msg_type):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if msg_type == "user":
            style = "background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin: 5px 0;"
            icon = "👤"
        else:
            style = "background-color: #f3e5f5; border-left: 4px solid #9c27b0; padding: 10px; margin: 5px 0;"
            icon = "🎯"
        
        html_content = f"""
        <div style="{style}">
            <strong>{icon} {sender}</strong> <small style="color: #666;">({timestamp})</small><br>
            <div style="margin-top: 5px; white-space: pre-wrap;">{message}</div>
        </div>
        """
        display(HTML(html_content))
    
    def send_message(self, button):
        user_input = self.input_text.value.strip()
        if not user_input:
            return
        
        # Show waiting message immediately
        with self.output:
            self.display_xml_message("Lyra", "🔄 Understanding and optimizing your prompt...", "assistant")
        
        # Capture stdout to prevent agent output from going to logs
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            response = self.optimizer.optimize_prompt(user_input)
        except Exception as e:
            response = f"Error: {str(e)}"
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Clear the waiting message and display actual response
        self.output.clear_output()
        with self.output:
            # Re-display welcome message if this is first interaction
            if len(self.chat_history) == 0:
                self.display_message("Lyra", self.optimizer.get_welcome_message(), "assistant")
            
            if response:
                self.display_xml_message("Lyra", response, "assistant")
            else:
                self.display_xml_message("Lyra", "No response received", "assistant")
        
        # Clear input
        self.input_text.value = ""
        
        # Store in history
        self.chat_history.append({"user": user_input, "assistant": response})
    
    def clear_chat(self, button):
        self.output.clear_output()
        self.chat_history = []
        with self.output:
            self.display_message("Lyra", self.optimizer.get_welcome_message(), "assistant")
    
    def display(self):
        # Display welcome message
        with self.output:
            self.display_message("Lyra", self.optimizer.get_welcome_message(), "assistant")
        
        # Configure output widget to remove scrolling
        self.output.layout = widgets.Layout(
            width='100%',
            height='auto',
            max_height='none',
            overflow='visible'
        )
        
        # Create layout with input at bottom
        chat_interface = widgets.VBox([
            widgets.HTML("<h4>💬 Chat with Lyra</h4>"),
            self.output,
            widgets.HTML("<hr>"),
            self.input_text,
            widgets.HBox([self.send_button, self.clear_button])
        ], layout=widgets.Layout(height='auto'))
        
        display(chat_interface)
