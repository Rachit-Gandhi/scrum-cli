#!/usr/bin/env python3
import subprocess
import time
import requests
import json
import logging
from typing import Optional, Dict
import threading
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class NgrokManager:
    """Manage ngrok tunnels for secure API proxy"""
    
    def __init__(self):
        self.process = None
        self.tunnel_url = None
        self.api_base = "http://127.0.0.1:4040"
        
    def start_tunnel(self, port: int = 8000, domain: Optional[str] = None) -> Optional[str]:
        """
        Start ngrok tunnel for the proxy server
        
        Args:
            port: Local port to tunnel
            domain: Optional ngrok domain (for paid accounts)
            
        Returns:
            Public tunnel URL or None if failed
        """
        try:
            # Build ngrok command
            cmd = ["ngrok", "http", str(port)]
            
            if domain:
                cmd.extend(["--domain", domain])
            
            # Add JSON logging for easier parsing
            cmd.extend(["--log", "stdout", "--log-format", "json"])
            
            logger.info(f"Starting ngrok tunnel: {' '.join(cmd)}")
            
            # Start ngrok process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give ngrok time to start
            time.sleep(3)
            
            # Get tunnel info from ngrok API
            tunnel_url = self._get_tunnel_url()
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                logger.info(f"Ngrok tunnel active: {tunnel_url}")
                return tunnel_url
            else:
                logger.error("Failed to get tunnel URL from ngrok")
                self.stop_tunnel()
                return None
                
        except FileNotFoundError:
            logger.error("ngrok not found. Please install ngrok: https://ngrok.com/download")
            return None
        except Exception as e:
            logger.error(f"Failed to start ngrok tunnel: {e}")
            return None
    
    def _get_tunnel_url(self) -> Optional[str]:
        """Get the public tunnel URL from ngrok API"""
        try:
            # Try to get tunnel info from ngrok local API
            response = requests.get(f"{self.api_base}/api/tunnels", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            tunnels = data.get("tunnels", [])
            
            for tunnel in tunnels:
                if tunnel.get("config", {}).get("addr") == "http://localhost:8000":
                    public_url = tunnel.get("public_url")
                    if public_url and public_url.startswith("https://"):
                        return public_url
            
            logger.warning("No HTTPS tunnel found in ngrok API response")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Failed to get tunnel info from ngrok API: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing ngrok API response: {e}")
            return None
    
    def stop_tunnel(self):
        """Stop the ngrok tunnel"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Ngrok tunnel stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Ngrok process didn't terminate gracefully, killing it")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping ngrok: {e}")
            finally:
                self.process = None
                self.tunnel_url = None
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get the current tunnel URL"""
        return self.tunnel_url
    
    def is_active(self) -> bool:
        """Check if tunnel is active"""
        return self.process is not None and self.process.poll() is None
    
    def get_status(self) -> Dict:
        """Get tunnel status info"""
        return {
            "active": self.is_active(),
            "url": self.tunnel_url,
            "process_id": self.process.pid if self.process else None
        }

def start_ngrok_tunnel(port: int = 8000, domain: Optional[str] = None) -> Optional[NgrokManager]:
    """
    Convenience function to start ngrok tunnel
    
    Args:
        port: Local port to tunnel
        domain: Optional ngrok domain
        
    Returns:
        NgrokManager instance or None if failed
    """
    manager = NgrokManager()
    tunnel_url = manager.start_tunnel(port, domain)
    
    if tunnel_url:
        return manager
    else:
        return None

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ngrok tunnel...")
    manager = start_ngrok_tunnel()
    
    if manager:
        print(f"Tunnel URL: {manager.get_tunnel_url()}")
        print("Press Enter to stop tunnel...")
        input()
        manager.stop_tunnel()
    else:
        print("Failed to start tunnel")