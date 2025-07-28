#!/usr/bin/env python3
"""
Standalone proxy server host script for running the centralized proxy server
Use this to host the proxy server that users can connect to
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from proxy_server import start_proxy_server
from ngrok_manager import start_ngrok_tunnel

def main():
    parser = argparse.ArgumentParser(description='Host SCRUMS-CLI Proxy Server')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--no-ngrok', action='store_true', help='Skip ngrok tunnel')
    parser.add_argument('--ngrok-domain', help='Custom ngrok domain')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check environment variables
    required_env = ['GEMINI_API_KEY']
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        logger.error(f"Missing required environment variables: {', '.join(missing_env)}")
        logger.info("Create a .env file with:")
        logger.info("GEMINI_API_KEY=your_gemini_api_key")
        logger.info("HUGGING_FACE_TOKEN=your_hf_token  # Optional for speaker diarization")
        sys.exit(1)
    
    # Show configuration
    logger.info("🚀 Starting SCRUMS-CLI Proxy Server Host")
    logger.info(f"Server: {args.host}:{args.port}")
    
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    if hf_token:
        logger.info("✅ HuggingFace token configured - speaker diarization enabled")
    else:
        logger.warning("⚠️ No HuggingFace token - speaker diarization disabled")
    
    try:
        # Start ngrok tunnel if requested
        ngrok_manager = None
        if not args.no_ngrok:
            logger.info("🌐 Starting ngrok tunnel...")
            ngrok_manager = start_ngrok_tunnel(args.port, args.ngrok_domain)
            
            if ngrok_manager:
                tunnel_url = ngrok_manager.get_tunnel_url()
                logger.info(f"✅ Ngrok tunnel active: {tunnel_url}")
                logger.info(f"📋 Users can connect with: --proxy-url {tunnel_url}")
            else:
                logger.warning("⚠️ Ngrok tunnel failed, running locally only")
        
        # Start the proxy server (this blocks)
        logger.info("🔥 Starting proxy server... (Ctrl+C to stop)")
        start_proxy_server(args.host, args.port)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down proxy server...")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        sys.exit(1)
    finally:
        if ngrok_manager:
            logger.info("🔒 Stopping ngrok tunnel...")
            ngrok_manager.stop_tunnel()
        logger.info("👋 Proxy server stopped")

if __name__ == "__main__":
    main()