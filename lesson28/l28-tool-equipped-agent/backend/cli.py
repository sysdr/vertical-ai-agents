#!/usr/bin/env python3
"""
Interactive CLI for L28 Tool-Equipped Agent
"""

import asyncio
import sys
from agent.tool_equipped_agent import ToolEquippedAgent
from tools.registry import ToolRegistry
from tools.weather import WeatherTool
from tools.calculator import CalculatorTool
from tools.database import DatabaseTool
from dotenv import load_dotenv
import json

load_dotenv()

class AgentCLI:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(WeatherTool())
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(DatabaseTool())
        
        self.agent = ToolEquippedAgent(tool_registry=self.tool_registry)
        self.conversation_id = None
    
    async def run(self):
        """Run interactive CLI"""
        print("=" * 60)
        print("L28 Tool-Equipped Agent - Interactive CLI")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  /tools  - List available tools")
        print("  /help   - Show this help")
        print("  /quit   - Exit")
        print("\nType your queries and press Enter...")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if not query:
                    continue
                
                if query == "/quit":
                    print("Goodbye!")
                    break
                
                if query == "/help":
                    self._show_help()
                    continue
                
                if query == "/tools":
                    self._show_tools()
                    continue
                
                # Process query
                print("\n[Processing...]")
                result = await self.agent.process_query(
                    query=query,
                    conversation_id=self.conversation_id
                )
                
                self.conversation_id = result["conversation_id"]
                
                # Display results
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def _show_help(self):
        print("\n📖 Help:")
        print("  - Ask factual questions (uses RAG)")
        print("  - Request calculations (uses calculator tool)")
        print("  - Check weather (uses weather tool)")
        print("  - Query database (uses database tool)")
        print("  - Complex queries use multiple systems")
    
    def _show_tools(self):
        print("\n🔧 Available Tools:")
        for tool in self.tool_registry.tools.values():
            print(f"  • {tool.name}: {tool.description}")
    
    def _display_result(self, result: dict):
        """Display formatted result"""
        print("\n" + "=" * 60)
        print(f"Intent: {result['intent_type'].upper()}")
        
        if result['tool_used']:
            print(f"Tool Used: {result['tool_used']}")
        
        print("\n📝 Response:")
        print(result['response'])
        
        if result.get('sources'):
            print(f"\n📚 Sources: {', '.join(result['sources'])}")
        
        if result.get('metrics'):
            print("\n📊 Evaluation Metrics:")
            metrics = result['metrics']
            print(f"  Groundedness: {metrics['groundedness_score']:.2f}")
            print(f"  Relevance:    {metrics['relevance_score']:.2f}")
            print(f"  Completeness: {metrics['completeness_score']:.2f}")
            print(f"  Overall:      {metrics['overall_score']:.2f}")
        
        print("=" * 60)

async def main():
    cli = AgentCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
