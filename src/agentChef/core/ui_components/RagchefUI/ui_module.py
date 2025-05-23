"""
ragchef UI Module - Simple graphical interface for the ragchef system.
Provides a user-friendly interface for research, dataset generation, and analysis.

Written By: @AgentChef
Date: 4/4/2025
"""

import os
import sys
import json
import asyncio
import threading
import random
import math
from pathlib import Path
import ollama

from datetime import datetime, timezone
UTC = timezone.utc

import logging
logger = logging.getLogger(__name__)
import webbrowser
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog,
        QProgressBar, QSpinBox, QCheckBox, QGroupBox, QFormLayout, QSplitter,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog, QSlider
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer, QPointF
    from PyQt6.QtGui import QFont, QIcon, QTextCursor, QPainter, QColor, QBrush, QPen
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6 not available. Install with 'pip install PyQt6'")

from agentChef.logs.agentchef_logging import log, setup_file_logging

class WorkerThread(QThread):
    """Worker thread for running asynchronous operations."""
    
    update_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        """
        Initialize the worker thread.
        
        Args:
            func: Asynchronous function to run
            *args, **kwargs: Arguments for the function
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        """Run the worker thread."""
        try:
            # Create a callback for progress updates
            def update_callback(message):
                self.update_signal.emit(message)
                
            # Add the callback to kwargs
            self.kwargs['callback'] = update_callback
            
            # Create and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.func(*self.args, **self.kwargs))
            self.result_signal.emit(result)
            
            loop.close()
            
        except Exception as e:
            logger.exception("Error in worker thread")
            self.error_signal.emit(f"Error: {str(e)}")

import random
import math
import numpy as np
from collections import defaultdict

class QuantumMeshParticle:
    """Quantum wave-based mesh particle for fluid dynamics."""
    
    def __init__(self, x, y, mesh_id=None):
        self.x = x
        self.y = y
        self.mesh_id = mesh_id or random.randint(1000, 9999)
        
        # Enhanced quantum wave properties for more fluid behavior
        self.wave_amplitude = random.uniform(2.0, 4.0)  # Increased amplitude
        self.wave_frequency = random.uniform(0.02, 0.08)  # More variation
        self.wave_phase = random.uniform(0, 2 * math.pi)
        self.wave_speed = random.uniform(0.03, 0.12)  # Faster waves for more fluid motion
        
        # Mesh connectivity - more connections for better fluidity
        self.connections = []
        self.influence_radius = random.uniform(20, 35)  # Larger radius for better connectivity
        self.mesh_strength = random.uniform(0.6, 0.9)  # More flexible connections
        
        # Enhanced physics properties for gooey behavior
        self.velocity_x = random.uniform(-0.08, 0.08)
        self.velocity_y = random.uniform(0.01, 0.12)
        self.mass = random.uniform(0.7, 1.3)  # More mass variation
        self.viscosity = random.uniform(0.85, 0.95)  # Lower viscosity for more fluid motion
        
        # Enhanced sticking properties
        self.is_stuck = False
        self.stick_timer = 0
        self.stick_strength = random.uniform(0.8, 1.0)
        self.detachment_threshold = random.uniform(0.3, 0.7)  # Easier detachment
        
        # Temperature and visual with more variation
        self.temperature = random.uniform(0.7, 1.0)
        self.glow_intensity = random.uniform(0.4, 1.0)
        self.fluidity = random.uniform(0.8, 1.0)  # New fluidity property

class FluidMeshDrop:
    """A fluid mesh drop with enhanced gooey physics."""
    
    def __init__(self, center_x, center_y, size=None, particle_density=1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.particle_density = particle_density  # Add this parameter
        # More random size variation for organic feel
        self.size = size or random.uniform(8, 35)  # Wider size range
        self.mesh_particles = []
        self.trail_particles = []
        self.detached_particles = []  # New: particles that detach during formation
        
        # Generate unique ID for this drop
        self.drop_id = random.randint(1000, 9999)
        
        # Enhanced raindrop physics for more fluid behavior
        self.velocity_x = random.uniform(-0.15, 0.15)
        self.velocity_y = random.uniform(0.01, 0.15)  # Slower initial formation
        self.acceleration_y = 0
        self.surface_tension = random.uniform(0.8, 1.5)  # Lower surface tension
        self.drag_coefficient = random.uniform(0.01, 0.025)
        
        # Enhanced mesh deformation for more organic movement
        self.deformation_phase = random.uniform(0, 2 * math.pi)
        self.deformation_speed = random.uniform(0.04, 0.15)  # Faster deformation
        self.jiggle_amplitude = random.uniform(1.2, 2.5)  # More jiggle
        
        # Enhanced gooey properties
        self.gooiness = random.uniform(0.7, 1.0)
        self.stretch_factor = 1.0
        self.is_stretching = False
        self.separation_threshold = random.uniform(30, 50)  # Distance before separation
        
        # Enhanced sticking properties
        self.ceiling_stick_time = random.uniform(120, 300)  # Longer stick time
        self.is_stuck_to_ceiling = False
        self.stick_timer = 0
        self.formation_timer = random.uniform(60, 180)  # Time to form before dropping
        
        # Visual properties with more variation
        self.temperature = random.uniform(0.6, 1.0)
        self.opacity = random.uniform(0.75, 1.0)
        self.blob_complexity = random.randint(8, 16)  # More complex shapes
        
        # Create enhanced mesh for better fluidity
        self._create_enhanced_mesh_structure()
    
    def get_id(self):
        """Get the unique ID for this drop."""
        return self.drop_id
    
    def _create_enhanced_mesh_structure(self):
        """Create enhanced mesh structure with more organic connectivity."""
        # Scale mesh points based on particle density
        base_body_points = max(6, int(self.size / 4))
        num_body_points = int(base_body_points * self.particle_density)
        num_body_points = max(3, min(num_body_points, 24))  # Clamp between 3 and 24
        
        for i in range(num_body_points):
            angle = (2 * math.pi * i) / num_body_points
            # More organic radius variation
            radius_var = random.uniform(0.3, 0.8)
            radius = self.size * radius_var
            
            # Add noise for organic shape
            angle_noise = random.uniform(-0.3, 0.3)
            radius_noise = random.uniform(-0.2, 0.2) * self.size
            
            x = self.center_x + math.cos(angle + angle_noise) * (radius + radius_noise)
            y = self.center_y + math.sin(angle + angle_noise) * (radius + radius_noise)
            
            particle = QuantumMeshParticle(x, y, self.get_id())
            particle.fluidity = self.gooiness
            self.mesh_particles.append(particle)
        
        # Enhanced trail particles with density scaling
        base_trail_length = max(3, int(self.size / 5))
        trail_length = int(base_trail_length * self.particle_density)
        trail_length = max(2, min(trail_length, 15))  # Clamp between 2 and 15
        
        for i in range(trail_length):
            trail_y = self.center_y - (i + 1) * random.uniform(1.5, 3.5)
            trail_width = self.size * (1.0 - (i / trail_length)) * random.uniform(0.2, 0.5)
            
            # Scale particles per segment based on density
            base_particles_per_segment = random.randint(1, 3)
            particles_per_segment = max(1, int(base_particles_per_segment * self.particle_density))
            
            for j in range(particles_per_segment):
                x = self.center_x + random.uniform(-trail_width, trail_width)
                y = trail_y + random.uniform(-1, 1)
                
                particle = QuantumMeshParticle(x, y, self.get_id())
                particle.fluidity = self.gooiness * random.uniform(0.8, 1.0)
                self.trail_particles.append(particle)
        
        # Enhanced connections with better connectivity
        self._create_enhanced_connections()
    
    def _create_enhanced_connections(self):
        """Create enhanced connections with better fluidity."""
        all_particles = self.mesh_particles + self.trail_particles
        
        for i, particle1 in enumerate(all_particles):
            connections_made = 0
            # Scale max connections based on density
            base_max_connections = random.randint(3, 6)
            max_connections = int(base_max_connections * self.particle_density)
            max_connections = max(2, min(max_connections, 10))  # Clamp between 2 and 10
            
            # Sort particles by distance for better connectivity
            distances = []
            for j, particle2 in enumerate(all_particles):
                if i != j:
                    distance = math.sqrt(
                        (particle2.x - particle1.x)**2 + 
                        (particle2.y - particle1.y)**2
                    )
                    distances.append((j, distance))
            
            distances.sort(key=lambda x: x[1])
            
            for j, distance in distances:
                if connections_made >= max_connections:
                    break
                    
                if distance < particle1.influence_radius:
                    particle1.connections.append(j)
                    all_particles[j].connections.append(i)
                    connections_made += 1

    def update_quantum_physics(self, time_step, gravity=0.1):
        """Update enhanced quantum mesh physics."""
        # Update deformation phase with more variation
        self.deformation_phase += self.deformation_speed
        
        # Handle formation timer for ceiling sticking
        if self.formation_timer > 0:
            self.formation_timer -= 1
            gravity *= 0.1  # Reduced gravity during formation
        
        # Apply gravity to drop center
        self.acceleration_y += gravity * random.uniform(0.8, 1.2)  # Gravity variation
        self.velocity_y += self.acceleration_y
        
        # Enhanced drag with more realistic behavior
        drag_factor = 1.0 - (self.drag_coefficient * (1 + abs(self.velocity_y) * 0.5))
        self.velocity_x *= drag_factor
        self.velocity_y *= drag_factor
        
        # Update center position
        self.center_x += self.velocity_x
        self.center_y += self.velocity_y
        
        # Update all mesh particles with enhanced quantum waves
        self._update_enhanced_mesh_particles(time_step)
        
        # Apply enhanced mesh connectivity forces
        self._apply_enhanced_mesh_forces()
        
        # Handle particle detachment for more organic behavior
        self._handle_particle_detachment()
        
        # Cool down temperature more gradually
        self.temperature = max(0.3, self.temperature - 0.001)

    def _update_enhanced_mesh_particles(self, time_step):
        """Update mesh particles with enhanced quantum wave behavior."""
        all_particles = self.mesh_particles + self.trail_particles
        
        for i, particle in enumerate(all_particles):
            # Enhanced quantum wave deformation
            particle.wave_phase += particle.wave_speed
            
            # Multi-layered wave interference with more complexity
            primary_wave = math.sin(particle.wave_phase) * particle.wave_amplitude
            secondary_wave = math.cos(particle.wave_phase * 1.7) * (particle.wave_amplitude * 0.8)
            tertiary_wave = math.sin(particle.wave_phase * 0.3 + self.deformation_phase) * (particle.wave_amplitude * 0.6)
            quaternary_wave = math.cos(particle.wave_phase * 2.3 + time_step * 0.02) * (particle.wave_amplitude * 0.4)
            
            total_deformation = (primary_wave + secondary_wave + tertiary_wave + quaternary_wave) * particle.fluidity
            
            # Apply wave deformation relative to drop center
            direction_x = particle.x - self.center_x
            direction_y = particle.y - self.center_y
            distance = max(1, math.sqrt(direction_x**2 + direction_y**2))
            
            # Normalize direction
            norm_x = direction_x / distance
            norm_y = direction_y / distance
            
            # Apply enhanced deformation
            deform_strength = 0.15 * particle.fluidity  # More deformation
            particle.x += norm_x * total_deformation * deform_strength
            particle.y += norm_y * total_deformation * deform_strength
            
            # Enhanced particle pulling with more organic behavior
            max_distance = self.size * (1.8 + particle.fluidity * 0.5)
            if distance > max_distance:
                pull_strength = (distance - max_distance) / max_distance
                pull_strength *= (1.0 + particle.fluidity)
                particle.x -= norm_x * pull_strength * 1.5
                particle.y -= norm_y * pull_strength * 1.5

    def _apply_enhanced_mesh_forces(self):
        """Apply enhanced connectivity forces between mesh particles."""
        all_particles = self.mesh_particles + self.trail_particles
        
        for i, particle in enumerate(all_particles):
            force_x = 0
            force_y = 0
            total_connections = len(particle.connections)
            
            # Apply forces from connected particles
            for j in particle.connections:
                if j < len(all_particles):
                    other = all_particles[j]
                    
                    # Calculate distance and direction
                    dx = other.x - particle.x
                    dy = other.y - particle.y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance > 0:
                        # Enhanced ideal connection distance based on fluidity
                        base_distance = (particle.influence_radius + other.influence_radius) / 4
                        fluidity_factor = (particle.fluidity + other.fluidity) * 0.5
                        ideal_distance = base_distance * (1.0 + fluidity_factor * 0.5)
                        
                        # Enhanced spring force with non-linear behavior
                        displacement = distance - ideal_distance
                        force_magnitude = displacement * 0.025 * fluidity_factor
                        
                        # Add damping for stability
                        relative_velocity_x = other.velocity_x - particle.velocity_x
                        relative_velocity_y = other.velocity_y - particle.velocity_y
                        damping_force_x = relative_velocity_x * 0.01
                        damping_force_y = relative_velocity_y * 0.01
                        
                        # Normalize direction and apply forces
                        force_x += (dx / distance) * force_magnitude + damping_force_x
                        force_y += (dy / distance) * force_magnitude + damping_force_y
            
            # Apply enhanced viscosity with fluidity consideration
            viscosity_factor = particle.viscosity * (1.0 - particle.fluidity * 0.2)
            particle.velocity_x = (particle.velocity_x + force_x) * viscosity_factor
            particle.velocity_y = (particle.velocity_y + force_y) * viscosity_factor
            
            # Update position
            particle.x += particle.velocity_x
            particle.y += particle.velocity_y

    def _handle_particle_detachment(self):
        """Handle particle detachment for more organic behavior."""
        all_particles = self.mesh_particles + self.trail_particles
        
        for i, particle in enumerate(all_particles):
            # Check if particle should detach
            distance_from_center = math.sqrt(
                (particle.x - self.center_x)**2 + 
                (particle.y - self.center_y)**2
            )
            
            # Detach if too far and low connectivity
            max_detach_distance = self.size * 2.5
            if (distance_from_center > max_detach_distance and 
                len(particle.connections) <= 1 and 
                random.random() < 0.02):  # Small chance of detachment
                
                # Create small detached droplet
                if particle in self.mesh_particles:
                    self.mesh_particles.remove(particle)
                elif particle in self.trail_particles:
                    self.trail_particles.remove(particle)
                
                # Remove connections
                for other_particle in all_particles:
                    if i in other_particle.connections:
                        other_particle.connections.remove(i)
                
                self.detached_particles.append(particle)

class QuantumFluidSystem(QWidget):
    """Enhanced quantum fluid mesh system with improved gooey physics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Widget setup
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # Enhanced quantum fluid system
        self.mesh_drops = []
        self.small_droplets = []  # New: small detached droplets
        self.ceiling_sources = []
        self.lava_pool = None
        self.physics_timer = None
        self.is_enabled = True
        self.time_step = 0
        
        # Particle density control (1.0 = default, 0.1 = minimal, 2.0 = maximum)
        self.particle_density = 1.0
        
        # Enhanced performance optimizations
        self.max_drops = 20  # Increased for more droplets
        self.max_small_droplets = 30  # Limit small droplets
        self.update_frequency = 30  # Slightly increased FPS
        
        # Enhanced physics constants
        self.gravity = 0.04  # Reduced for slower formation
        self.max_gravity = 0.15
        self.quantum_coherence = 0.85
        self.mesh_fluidity = 0.95
        
        # Enhanced pool settings
        self.pool_fill_rate = 1.5  # Faster filling
        self.pool_drain_rate = 2.0
        self.is_draining = False
        
        self.setVisible(True)
        self.raise_()
        
        QTimer.singleShot(100, self._delayed_init)

    def set_particle_density(self, density):
        """Set the particle density level."""
        self.particle_density = max(0.1, min(2.0, density))  # Clamp between 0.1 and 2.0
        
        # Scale ceiling sources based on density
        base_num_sources = 4
        scaled_sources = int(base_num_sources * self.particle_density)
        scaled_sources = max(2, min(scaled_sources, 10))  # Clamp between 2 and 10
        
        # If density changed significantly, recreate sources
        if abs(len(self.ceiling_sources) - scaled_sources) > 1:
            self._create_ceiling_sources()
            
    def _delayed_init(self):
        """Initialize the quantum fluid system."""
        if self.width() <= 0 or self.height() <= 0:
            self.resize(1000, 700)
            
        self.lava_pool = LavaPool(self.width(), self.height())
        self._create_ceiling_sources()
        
        self.physics_timer = QTimer()
        self.physics_timer.timeout.connect(self.update_quantum_physics)
        self.physics_timer.start(int(1000 / self.update_frequency))  # Optimized FPS
        
        logger.info(f"Quantum fluid system initialized: {self.width()}x{self.height()}")

    def _create_ceiling_sources(self):
        """Create enhanced quantum mesh sources with more variation."""
        width = max(self.width(), 800)
        self.ceiling_sources = []
        
        # Scale number of sources based on particle density
        base_num_sources = random.randint(4, 7)
        num_sources = int(base_num_sources * self.particle_density)
        num_sources = max(2, min(num_sources, 12))  # Clamp between 2 and 12
        
        for i in range(num_sources):
            x_base = (width / (num_sources + 1)) * (i + 1)
            x = x_base + random.uniform(-80, 80)  # More spread
            x = max(60, min(width - 60, x))
            
            # Scale source size based on density
            base_size = random.uniform(30, 60)
            source_size = base_size * (0.7 + 0.3 * self.particle_density)  # Size scales with density
            
            source = {
                'x': x,
                'y': random.uniform(-8, 8),
                'size': source_size,
                'quantum_energy': random.uniform(300, 500),  # More energy
                'emission_timer': random.uniform(0, 200),
                'emission_interval': random.uniform(120, 400),  # More random intervals
                'coherence_level': random.uniform(0.7, 1.0),  # More variation
                'mesh_complexity': random.randint(8, 16),  # More complex
                'temperature': random.uniform(0.8, 1.0),  # More temperature variation
                # Enhanced visual properties
                'glow_phase': random.uniform(0, 2 * math.pi),
                'glow_speed': random.uniform(0.015, 0.08),  # More variation
                'pulsation': random.uniform(0.7, 1.3),  # More pulsation
                'heat_distortion': random.uniform(0.3, 2.0),  # More distortion
                'formation_speed': random.uniform(0.5, 1.5)  # Speed of drop formation
            }
            self.ceiling_sources.append(source)

    def _create_quantum_mesh_drop(self, source):
        """Create drops with enhanced sticking and formation behavior."""
        # More size variation based on source properties
        base_size = random.uniform(12, 30)
        size_multiplier = source['coherence_level'] * source['formation_speed']
        drop_size = base_size * size_multiplier
        
        # Starting position with more variation
        start_x = source['x'] + random.uniform(-15, 15)
        start_y = source['y'] + source['size'] * random.uniform(0.6, 0.9)
        
        # Create enhanced mesh drop with particle density
        drop = FluidMeshDrop(start_x, start_y, drop_size, self.particle_density)
        drop.temperature = source['temperature']
        
        # Enhanced sticking behavior with longer formation time
        drop.is_stuck_to_ceiling = True
        drop.stick_timer = drop.ceiling_stick_time * random.uniform(1.2, 2.0)  # Longer sticking
        drop.formation_timer = random.uniform(80, 250)  # Formation time
        drop.velocity_y = 0
        
        # Enhanced quantum properties
        drop.jiggle_amplitude *= source['coherence_level'] * 1.2
        drop.surface_tension *= source['coherence_level']
        drop.gooiness = source['coherence_level'] * random.uniform(0.9, 1.0)
        
        self.mesh_drops.append(drop)

    def update_quantum_physics(self):
        """Enhanced physics update with improved fluid behavior."""
        if not self.is_enabled:
            return
            
        self.time_step += 1
        width = max(self.width(), 800)
        height = max(self.height(), 600)
        
        # Update ceiling sources with more variation
        if self.time_step % random.randint(1, 3) == 0:
            for source in self.ceiling_sources:
                self._update_enhanced_ceiling_source(source, width)
        
        # Update existing mesh drops with enhanced physics
        for drop in self.mesh_drops[:]:
            self._update_enhanced_drop_physics(drop, height)
            
            # Handle drop reaching pool with splash effects
            if drop.center_y > height - self.lava_pool.level - 20:
                self._create_splash_effect(drop)
                self.lava_pool.add_lava(self.pool_fill_rate * drop.size * 0.15)
                self.mesh_drops.remove(drop)
            elif drop.center_x < -80 or drop.center_x > width + 80:
                self.mesh_drops.remove(drop)
        
        # Update small droplets
        self._update_small_droplets(height)
        
        # Apply enhanced quantum coherence
        if self.time_step % 2 == 0:  # More frequent updates
            self._apply_enhanced_gooey_coherence()
        
        # Update lava pool with enhanced behavior
        self.lava_pool.update()
        
        # Handle draining
        if self.is_draining:
            self.lava_pool.drain(self.pool_drain_rate)
        
        # Limit drops for performance
        if len(self.mesh_drops) > self.max_drops:
            self.mesh_drops = self.mesh_drops[-self.max_drops:]
        
        if len(self.small_droplets) > self.max_small_droplets:
            self.small_droplets = self.small_droplets[-self.max_small_droplets:]
        
        self.update()

    def _update_drop_physics(self, drop, height):
        """Update individual drop with sticking and gooey physics."""
        # Handle ceiling sticking
        if drop.is_stuck_to_ceiling:
            drop.stick_timer -= 1
            
            # Add slight jiggle while stuck
            drop.center_x += math.sin(self.time_step * 0.05 + drop.center_x * 0.01) * 0.2
            
            # Release when timer expires or if too stretched
            if drop.stick_timer <= 0 or drop.stretch_factor > 1.5:
                drop.is_stuck_to_ceiling = False
                drop.velocity_y = random.uniform(0.02, 0.08)  # Gentle release
        else:
            # Normal physics when not stuck
            drop.update_quantum_physics(self.time_step, self.gravity)
            
            # Check for nearby ceiling particles to re-stick
            self._check_ceiling_sticking(drop)

    def _check_ceiling_sticking(self, drop):
        """Check if drop should stick to nearby ceiling particles."""
        if drop.center_y > 100:  # Too far from ceiling
            return
            
        # Check if close to any ceiling source
        for source in self.ceiling_sources:
            distance = math.sqrt(
                (drop.center_x - source['x'])**2 + 
                (drop.center_y - source['y'])**2
            )
            
            if distance < source['size'] * 1.2:  # Within sticking range
                # Stick if moving slowly enough
                if abs(drop.velocity_y) < 0.1:
                    drop.is_stuck_to_ceiling = True
                    drop.stick_timer = random.uniform(30, 90)
                    drop.velocity_y = 0
                    break

    def _apply_gooey_coherence(self):
        """Apply enhanced gooey coherence between drops."""
        for i, drop1 in enumerate(self.mesh_drops):
            for j, drop2 in enumerate(self.mesh_drops[i+1:], i+1):
                distance = math.sqrt(
                    (drop2.center_x - drop1.center_x)**2 + 
                    (drop2.center_y - drop1.center_y)**2
                )
                
                # Gooey connection range
                connection_range = 40 + (drop1.gooiness + drop2.gooiness) * 20
                
                if distance < connection_range and distance > 0:
                    # Stronger gooey force
                    gooey_strength = (connection_range - distance) / connection_range
                    gooey_strength *= (drop1.gooiness + drop2.gooiness) * 0.5
                    
                    # Apply stretchy attractive force
                    force_multiplier = 0.02 * gooey_strength
                    force_x = ((drop2.center_x - drop1.center_x) / distance) * force_multiplier
                    force_y = ((drop2.center_y - drop1.center_y) / distance) * force_multiplier
                    
                    # Only apply if not stuck to ceiling
                    if not drop1.is_stuck_to_ceiling:
                        drop1.velocity_x += force_x * 0.8
                        drop1.velocity_y += force_y * 0.8
                    if not drop2.is_stuck_to_ceiling:
                        drop2.velocity_x -= force_x * 0.8
                        drop2.velocity_y -= force_y * 0.8
                    
                    # Mark as stretching for visual effects
                    if distance < connection_range * 0.6:
                        drop1.is_stretching = True
                        drop2.is_stretching = True
                        drop1.stretch_factor = max(1.0, distance / 20)
                        drop2.stretch_factor = max(1.0, distance / 20)

    def _update_ceiling_source(self, source, width):
        """Update ceiling quantum source."""
        source['emission_timer'] += 1
        
        # Create new mesh drop
        if (source['emission_timer'] >= source['emission_interval'] and 
            source['quantum_energy'] > 30):
            
            self._create_quantum_mesh_drop(source)
            source['emission_timer'] = 0
            source['emission_interval'] = random.uniform(80, 350)
            source['quantum_energy'] -= random.uniform(15, 25)
        
        # Recharge quantum energy
        if random.random() < 0.02:
            source['quantum_energy'] += random.uniform(3, 8)
            source['quantum_energy'] = min(source['quantum_energy'], 300)

    def _draw_quantum_source(self, painter, source):
        """Draw enhanced realistic quantum source."""
        # Check if painter is still active
        if not painter.isActive():
            return
            
        # Update visual properties
        source['glow_phase'] += source['glow_speed']
        
        # Energy-based colors with more variation
        energy_ratio = source['quantum_energy'] / 400.0
        heat_intensity = 0.5 + 0.5 * math.sin(source['glow_phase'])
        
        try:
            # Multiple layered effects
            # Heat distortion background
            distortion_size = source['size'] * (1.5 + source['heat_distortion'] * heat_intensity)
            distortion_color = QColor(255, 100, 0, int(40 * energy_ratio))
            painter.setBrush(QBrush(distortion_color))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.drawEllipse(
                int(source['x'] - distortion_size/2),
                int(source['y'] - distortion_size/4),
                int(distortion_size),
                int(distortion_size * 0.3)
            )
            
            # Main molten core with pulsation
            core_size = source['size'] * source['pulsation'] * (0.9 + 0.1 * heat_intensity)
            
            # Create gradient effect manually by drawing multiple ellipses
            for layer in range(3):
                if not painter.isActive():
                    break
                    
                layer_size = core_size * (1.0 - layer * 0.2)
                layer_alpha = int((220 - layer * 40) * energy_ratio)
                layer_color = QColor(255, 60 + layer * 30, layer * 20, layer_alpha)
                
                painter.setBrush(QBrush(layer_color))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(
                    int(source['x'] - layer_size/2),
                    int(source['y'] - layer_size/4),
                    int(layer_size),
                    int(layer_size * 0.5)
                )
            
            # Bright center core
            if painter.isActive():
                center_size = core_size * 0.3
                center_color = QColor(255, 255, 200, int(255 * energy_ratio))
                painter.setBrush(QBrush(center_color))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(
                    int(source['x'] - center_size/2),
                    int(source['y'] - center_size/6),
                    int(center_size),
                    int(center_size * 0.3)
                )
        except Exception as e:
            logger.error(f"Error drawing quantum source: {e}")

    def _draw_quantum_mesh_drop(self, painter, drop):
        """Draw gooey quantum mesh drop with metaball effect."""
        if not painter.isActive():
            return
            
        if not drop.mesh_particles and not drop.trail_particles:
            return
        
        try:
            # Temperature-based color with gooey enhancement
            temp_color = self._get_gooey_temperature_color(drop.temperature, drop.opacity, drop.gooiness)
            
            # Draw main gooey blob using metaball technique
            self._draw_gooey_metaball(painter, drop, temp_color)
            
            # Draw stretching effects if applicable
            if drop.is_stretching and drop.stretch_factor > 1.2:
                self._draw_stretch_effects(painter, drop, temp_color)
        except Exception as e:
            logger.error(f"Error drawing quantum mesh drop: {e}")

    def _ensure_initialized(self):
        """Ensure the system is properly initialized."""
        if not self.ceiling_sources and self.width() > 0 and self.height() > 0:
            self._create_ceiling_sources()
            
    def paintEvent(self, event):
        """Paint the quantum fluid mesh system."""
        painter = QPainter()
        
        # Check if painter can begin painting
        if not painter.begin(self):
            return
        
        # Verify widget is ready for painting
        if self.width() <= 0 or self.height() <= 0 or not self.isVisible():
            painter.end()
            return
        
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
            
            # Check if we have valid data to paint
            if not hasattr(self, 'lava_pool') or not hasattr(self, 'ceiling_sources'):
                return
            
            # Draw lava pool first
            if self.lava_pool and self.lava_pool.level > 0:
                self._draw_lava_pool(painter)
            
            # Draw ceiling sources
            for source in self.ceiling_sources:
                if painter.isActive():
                    self._draw_quantum_source(painter, source)
            
            # Draw quantum mesh drops
            for drop in self.mesh_drops:
                if painter.isActive():
                    self._draw_quantum_mesh_drop(painter, drop)
            
            # Draw small droplets
            for droplet in self.small_droplets:
                if painter.isActive():
                    self._draw_small_droplet(painter, droplet)
                    
        except Exception as e:
            logger.error(f"Error in quantum fluid painting: {e}")
        finally:
            # Always end the painter
            if painter.isActive():
                painter.end()
            
    def _draw_small_droplet(self, painter, droplet):
        """Draw small detached droplets."""
        if not painter.isActive():
            return
        
        try:
            # Get droplet color based on temperature
            temp_color = self._get_gooey_temperature_color(
                droplet.temperature, 
                droplet.opacity, 
                0.5  # Lower gooiness for small droplets
            )
            
            painter.setBrush(QBrush(temp_color))
            painter.setPen(QPen(temp_color, 1))
            
            # Draw simple circle
            radius = droplet.size / 2
            painter.drawEllipse(
                int(droplet.center_x - radius),
                int(droplet.center_y - radius),
                int(droplet.size),
                int(droplet.size)
            )
            
        except Exception as e:
            logger.error(f"Error drawing small droplet: {e}")
        
    def _draw_gooey_metaball(self, painter, drop, temp_color):
        """Draw smooth gooey metaball shape."""
        if not painter.isActive():
            return
            
        try:
            painter.setBrush(QBrush(temp_color))
            painter.setPen(QPen(temp_color, 1))
            
            # Main body - more organic shape
            main_radius = drop.size * 0.8
            deform = math.sin(drop.deformation_phase) * drop.jiggle_amplitude * 0.5
            
            # Create organic blob points
            blob_points = []
            num_points = 12  # Smooth curve
            
            for i in range(num_points):
                angle = (2 * math.pi * i) / num_points
                
                # Organic radius variation
                radius_var = 0.8 + 0.2 * math.sin(angle * 3 + drop.deformation_phase)
                radius_var += 0.1 * math.sin(angle * 5 + drop.deformation_phase * 1.5)
                radius = main_radius * radius_var
                
                # Add gooey deformation
                radius += deform * 0.2 * math.sin(angle * 2)
                
                # Flatten if stuck to ceiling
                y_scale = 0.7 if drop.is_stuck_to_ceiling else 1.0
                
                x = drop.center_x + math.cos(angle) * radius
                y = drop.center_y + math.sin(angle) * radius * y_scale
                
                blob_points.append(QPointF(x, y))
            
            # Draw smooth blob
            if painter.isActive() and len(blob_points) >= 3:
                painter.drawPolygon(blob_points)
            
            # Add tail if not stuck to ceiling
            if not drop.is_stuck_to_ceiling and painter.isActive():
                self._draw_gooey_tail(painter, drop, temp_color, main_radius)
        except Exception as e:
            logger.error(f"Error drawing gooey metaball: {e}")

    def _draw_gooey_tail(self, painter, drop, temp_color, main_radius):
        """Draw gooey dripping tail."""
        if not painter.isActive():
            return
            
        try:
            tail_length = drop.size * 1.2
            tail_segments = 8
            
            tail_points = []
            
            for i in range(tail_segments):
                progress = i / (tail_segments - 1)
                y_offset = -tail_length * progress
                
                # Tail width tapering
                width = main_radius * 0.4 * (1.0 - progress * 0.8)
                
                # Add gooey wiggle
                wiggle = math.sin(drop.deformation_phase + progress * 3) * drop.jiggle_amplitude * 0.2
                
                # Create tail points
                left_x = drop.center_x - width + wiggle
                right_x = drop.center_x + width + wiggle
                y = drop.center_y + y_offset
                
                tail_points.append(QPointF(left_x, y))
                tail_points.insert(0, QPointF(right_x, y))  # Mirror for smooth shape
            
            # Draw tail with slightly transparent color
            if painter.isActive():
                tail_color = QColor(temp_color.red(), temp_color.green(), temp_color.blue(), int(temp_color.alpha() * 0.8))
                painter.setBrush(QBrush(tail_color))
                painter.setPen(QPen(tail_color, 1))
                
                if len(tail_points) >= 3:
                    painter.drawPolygon(tail_points)
        except Exception as e:
            logger.error(f"Error drawing gooey tail: {e}")

    def _draw_stretch_effects(self, painter, drop, temp_color):
        """Draw stretching gooey effects between connected drops."""
        # Find nearby drops for stretching effects
        for other_drop in self.mesh_drops:
            if other_drop == drop:
                continue
                
            distance = math.sqrt(
                (other_drop.center_x - drop.center_x)**2 + 
                (other_drop.center_y - drop.center_y)**2
            )
            
            if distance < 60 and distance > 10:  # Stretching range
                # Draw gooey connection
                stretch_color = QColor(
                    temp_color.red(), 
                    temp_color.green(), 
                    temp_color.blue(), 
                    int(temp_color.alpha() * 0.4)
                )
                
                connection_width = max(2, int(drop.size * 0.2 * (60 - distance) / 60))
                painter.setPen(QPen(stretch_color, connection_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(
                    int(drop.center_x), int(drop.center_y),
                    int(other_drop.center_x), int(other_drop.center_y)
                )

    def _get_gooey_temperature_color(self, temperature, opacity, gooiness):
        """Get enhanced gooey temperature color."""
        # Base temperature colors
        if temperature > 0.9:
            base_color = QColor(255, 23, 68)    # Hot
        elif temperature > 0.7:
            base_color = QColor(255, 69, 0)     # Warm
        elif temperature > 0.5:
            base_color = QColor(220, 20, 60)    # Medium
        elif temperature > 0.3:
            base_color = QColor(178, 34, 34)    # Cool
        else:
            base_color = QColor(139, 69, 19)    # Cold
        
        # Enhance with gooiness
        gooey_enhancement = int(gooiness * 50)
        final_color = QColor(
            min(255, base_color.red() + gooey_enhancement),
            base_color.green(),
            base_color.blue(),
            int(opacity * 255)
        )
        
        return final_color

    def _draw_lava_pool(self, painter):
        """Draw lava pool with quantum surface effects."""
        if not painter.isActive():
            return
            
        pool = self.lava_pool
        
        if not pool or pool.level <= 0:
            return
        
        try:
            # Pool body
            pool_color = QColor(255, 23, 68, 200)
            painter.setBrush(QBrush(pool_color))
            painter.setPen(QPen(pool_color, 0))
            
            # Draw pool rectangle
            if painter.isActive():
                painter.drawRect(
                    0, 
                    int(self.height() - pool.level), 
                    int(self.width()), 
                    int(pool.level)
                )
            
            # Quantum surface waves
            if painter.isActive():
                wave_points = []
                for x in range(0, int(self.width()), 3):
                    # Multi-frequency quantum waves
                    primary = math.sin(x * 0.02 + self.time_step * 0.08) * 4
                    secondary = math.cos(x * 0.05 + self.time_step * 0.12) * 2
                    tertiary = math.sin(x * 0.01 + self.time_step * 0.05) * 6
                    
                    wave_height = primary + secondary + tertiary
                    y = self.height() - pool.level + wave_height
                    wave_points.append(QPointF(x, y))
                
                # Draw quantum surface
                if wave_points and painter.isActive():
                    surface_color = QColor(255, 100, 100, 180)
                    painter.setPen(QPen(surface_color, 3))
                    
                    for i in range(len(wave_points) - 1):
                        if not painter.isActive():
                            break
                        painter.drawLine(wave_points[i], wave_points[i + 1])
        except Exception as e:
            logger.error(f"Error drawing lava pool: {e}")

    def toggle_particles(self):
        """Toggle the quantum fluid system."""
        self.is_enabled = not self.is_enabled
        if not self.is_enabled:
            # Clear all particle systems
            self.mesh_drops.clear()
            self.small_droplets.clear()  # Add this line
            self.ceiling_sources.clear()
            
            # Reset lava pool
            if hasattr(self, 'lava_pool') and self.lava_pool:
                self.lava_pool.level = 0
                self.lava_pool.surface_particles.clear()
                self.lava_pool.bubbles.clear()
            
            # Stop physics timer
            if hasattr(self, 'physics_timer') and self.physics_timer:
                self.physics_timer.stop()
            
            self.update()
        else:
            # Re-enable the system
            self._create_ceiling_sources()
            
            # Recreate lava pool
            if hasattr(self, 'lava_pool'):
                self.lava_pool = LavaPool(self.width(), self.height())
            
            # Restart physics timer
            if not hasattr(self, 'physics_timer') or not self.physics_timer:
                self.physics_timer = QTimer()
                self.physics_timer.timeout.connect(self.update_quantum_physics)
            
            self.physics_timer.start(int(1000 / self.update_frequency))

    def toggle_drain(self):
        """Toggle pool draining."""
        self.is_draining = not self.is_draining

    def resizeEvent(self, event):
        """Handle resize with quantum field adjustment."""
        super().resizeEvent(event)
        if self.lava_pool:
            # Scale pool level to new height
            old_height = self.lava_pool.height
            self.lava_pool.width = self.width()
            self.lava_pool.height = self.height()
            
            if old_height > 0:
                scale_factor = self.height() / old_height
                self.lava_pool.level *= scale_factor
                self.lava_pool.max_level = self.height() * 0.8
        
        if self.ceiling_sources:
            self._create_ceiling_sources()

    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        # Small delay before creating sources to ensure widget is ready
        QTimer.singleShot(50, self._ensure_initialized)

    def _update_small_droplets(self, height):
        """Update small detached droplets."""
        for droplet in self.small_droplets[:]:
            # Update droplet physics
            droplet.velocity_y += 0.02  # Light gravity
            droplet.center_x += droplet.velocity_x
            droplet.center_y += droplet.velocity_y
            
            # Add some drift
            droplet.velocity_x += random.uniform(-0.005, 0.005)
            droplet.velocity_y += random.uniform(-0.002, 0.002)
            
            # Apply drag
            droplet.velocity_x *= 0.98
            droplet.velocity_y *= 0.98
            
            # Remove if out of bounds or reached pool
            if (droplet.center_y > height - (self.lava_pool.level if self.lava_pool else 0) - 10 or
                droplet.center_x < -50 or droplet.center_x > self.width() + 50):
                
                if droplet in self.small_droplets:
                    self.small_droplets.remove(droplet)
                    
                    # Add to pool if it reached the bottom
                    if droplet.center_y > height - (self.lava_pool.level if self.lava_pool else 0) - 10:
                        if self.lava_pool:
                            self.lava_pool.add_lava(0.5)  # Small amount

    def _create_splash_effect(self, drop):
        """Create splash effect when drop hits pool."""
        # Create small droplets from the splash
        num_splash_droplets = random.randint(2, 5)
        
        for _ in range(num_splash_droplets):
            # Create small droplet with splash physics
            splash_droplet = type('SmallDroplet', (), {
                'center_x': drop.center_x + random.uniform(-15, 15),
                'center_y': drop.center_y - random.uniform(5, 15),
                'size': random.uniform(3, 8),
                'velocity_x': random.uniform(-0.2, 0.2),
                'velocity_y': random.uniform(-0.1, -0.05),  # Upward splash
                'temperature': drop.temperature * 0.8,
                'opacity': 0.7
            })()
            
            self.small_droplets.append(splash_droplet)
            
        # Limit number of small droplets
        if len(self.small_droplets) > self.max_small_droplets:
            self.small_droplets = self.small_droplets[-self.max_small_droplets:]

    def _update_enhanced_ceiling_source(self, source, width):
        """Update enhanced ceiling quantum source."""
        source['emission_timer'] += 1
        
        # Update visual properties
        source['glow_phase'] += source['glow_speed']
        source['pulsation'] = 0.9 + 0.1 * math.sin(source['glow_phase'])
        
        # Create new mesh drop
        if (source['emission_timer'] >= source['emission_interval'] and 
            source['quantum_energy'] > 30):
            
            self._create_quantum_mesh_drop(source)
            source['emission_timer'] = 0
            source['emission_interval'] = random.uniform(120, 400)
            source['quantum_energy'] -= random.uniform(15, 25)
        
        # Recharge quantum energy
        if random.random() < 0.02:
            source['quantum_energy'] += random.uniform(3, 8)
            source['quantum_energy'] = min(source['quantum_energy'], 500)

    def _update_enhanced_drop_physics(self, drop, height):
        """Update individual drop with enhanced sticking and gooey physics."""
        # Handle ceiling sticking
        if drop.is_stuck_to_ceiling:
            drop.stick_timer -= 1
            
            # Add slight jiggle while stuck
            drop.center_x += math.sin(self.time_step * 0.05 + drop.center_x * 0.01) * 0.2
            
            # Release when timer expires or if too stretched
            if drop.stick_timer <= 0 or drop.stretch_factor > 1.5:
                drop.is_stuck_to_ceiling = False
                drop.velocity_y = random.uniform(0.02, 0.08)  # Gentle release
        else:
            # Normal physics when not stuck
            drop.update_quantum_physics(self.time_step, self.gravity)
            
            # Check for nearby ceiling particles to re-stick
            self._check_ceiling_sticking(drop)

    def _apply_enhanced_gooey_coherence(self):
        """Apply enhanced gooey coherence between drops."""
        for i, drop1 in enumerate(self.mesh_drops):
            for j, drop2 in enumerate(self.mesh_drops[i+1:], i+1):
                distance = math.sqrt(
                    (drop2.center_x - drop1.center_x)**2 + 
                    (drop2.center_y - drop1.center_y)**2
                )
                
                # Enhanced gooey connection range
                base_range = 45
                gooey_bonus = (drop1.gooiness + drop2.gooiness) * 15
                connection_range = base_range + gooey_bonus
                
                if distance < connection_range and distance > 0:
                    # Stronger gooey force with non-linear behavior
                    gooey_strength = (connection_range - distance) / connection_range
                    gooey_strength = gooey_strength ** 1.5  # Non-linear for more dramatic effect
                    gooey_strength *= (drop1.gooiness + drop2.gooiness) * 0.6
                    
                    # Apply stretchy attractive force
                    force_multiplier = 0.03 * gooey_strength
                    force_x = ((drop2.center_x - drop1.center_x) / distance) * force_multiplier
                    force_y = ((drop2.center_y - drop1.center_y) / distance) * force_multiplier
                    
                    # Only apply if not stuck to ceiling
                    if not drop1.is_stuck_to_ceiling:
                        drop1.velocity_x += force_x
                        drop1.velocity_y += force_y * 0.8  # Slightly less vertical force
                    if not drop2.is_stuck_to_ceiling:
                        drop2.velocity_x -= force_x
                        drop2.velocity_y -= force_y * 0.8
                    
                    # Mark as stretching for visual effects
                    if distance < connection_range * 0.7:
                        drop1.is_stretching = True
                        drop2.is_stretching = True
                        drop1.stretch_factor = max(1.0, distance / 25)
                        drop2.stretch_factor = max(1.0, distance / 25)
                    
class LavaPool:
    """Pool of lava at the bottom with bubbling physics."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.level = 0  # How high the pool is
        self.max_level = height * 0.8  # Don't fill entire screen
        self.particles = []
        self.bubbles = []
        self.surface_particles = []
        self.time = 0
        
    def add_lava(self, amount):
        """Add lava to the pool."""
        if amount <= 0:
            return
            
        self.level = min(self.level + amount, self.max_level)
        self.level = max(0, self.level)  # Ensure never negative
        
        # Create surface particles for new width
        if self.level > 5:  # Only create particles if pool has meaningful depth
            num_surface = int(max(1, self.width / 20))
            for i in range(num_surface):
                if len(self.surface_particles) < num_surface:
                    x = (i + 0.5) * (self.width / num_surface)
                    y = self.height - self.level
                    particle = QuantumMeshParticle(x, y, random.randint(5000, 6000))
                    particle.velocity_y = 0
                    self.surface_particles.append(particle)
    
    def create_bubble(self):
        """Create a bubble in the pool."""
        if self.level > 20:
            x = random.uniform(20, self.width - 20)
            y = self.height - random.uniform(5, self.level - 10)
            
            bubble = {
                'x': x,
                'y': y,
                'size': random.uniform(4, 12),
                'vy': -random.uniform(0.5, 2.0),
                'vx': random.uniform(-0.2, 0.2),
                'life': random.uniform(60, 120),
                'pop_height': self.height - self.level + random.uniform(0, 10),
                'jiggle_phase': random.uniform(0, 2 * math.pi),
                'jiggle_speed': random.uniform(0.1, 0.3)
            }
            self.bubbles.append(bubble)
    
    def update(self):
        """Update pool physics."""
        self.time += 1
        
        # Create bubbles randomly
        if random.random() < 0.1 and self.level > 20:
            self.create_bubble()
        
        # Update surface particles
        for particle in self.surface_particles:
            particle.wave_phase += particle.wave_speed
            
            # Surface wave motion
            wave_height = math.sin(particle.x * 0.02 + self.time * 0.05) * 3
            particle.y = self.height - self.level + wave_height
            
            # Temperature cooling
            particle.temperature = max(0.3, particle.temperature - 0.001)
        
        # Update bubbles
        for bubble in self.bubbles[:]:
            bubble['jiggle_phase'] += bubble['jiggle_speed']
            
            # Jiggle motion
            jiggle_x = math.sin(bubble['jiggle_phase']) * 1.5
            jiggle_y = math.cos(bubble['jiggle_phase'] * 1.5) * 0.8
            
            bubble['x'] += bubble['vx'] + jiggle_x * 0.1
            bubble['y'] += bubble['vy'] + jiggle_y * 0.1
            
            # Buoyancy
            bubble['vy'] -= 0.02
            bubble['life'] -= 1
            
            # Remove if popped or died
            if bubble['y'] <= bubble['pop_height'] or bubble['life'] <= 0:
                self.bubbles.remove(bubble)
    
    def drain(self, amount):
        """Drain lava from the pool."""
        if amount <= 0:
            return
            
        self.level = max(0, self.level - amount)
        
        # Remove surface particles if pool is empty or very low
        if self.level <= 5:
            self.surface_particles.clear()
            self.bubbles.clear()

class ConversationViewer(QWidget):
    """Widget for displaying generated conversations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Conversation display with transparency
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(31, 41, 55, 0.85);  /* Dark background with 85% opacity */
                color: #FFFFFF;
                border: 1px solid rgba(55, 65, 81, 0.8);  /* Semi-transparent border */
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        self.layout.addWidget(self.display)

    def add_message(self, role: str, content: str):
        """Add a new message to the conversation."""
        color = "#FF1744" if role == "human" else "#00E5FF"
        self.display.append(f'<p style="margin: 4px 0;"><span style="color: {color}">{role}:</span> {content}</p>')

class RagchefUI(QMainWindow):
    """Main window for the ragchef UI."""
    
    def __init__(self, research_manager):
        """Initialize the UI."""
        super().__init__()
        self.research_manager = research_manager
        self.worker_thread = None
        
        # Set up logging for UI
        setup_file_logging("./logs/ui")
        logger.info("Initializing RagchefUI...")
        
        self.setWindowTitle("ragchef - Unified Dataset Research, Augmentation, & Generation System")
        self.setMinimumSize(1000, 700)
        
        # Set up the UI first
        self._setup_ui()
        
        # Add quantum fluid system
        self.particles = QuantumFluidSystem(self)
        self.particles.setGeometry(0, 0, self.width(), self.height())
        self.particles.raise_()
        self.particles.show()
        
        # Create control buttons and slider
        button_y = 10
        button_spacing = 50
        
        # Particle density slider
        self.density_slider = QSlider(Qt.Orientation.Vertical, self)
        self.density_slider.setRange(10, 200)  # 0.1 to 2.0 (multiplied by 100)
        self.density_slider.setValue(100)  # Default 1.0
        self.density_slider.setFixedSize(30, 150)
        self.density_slider.setToolTip("Particle Density\n(Low ← → High)")
        self.density_slider.setStyleSheet("""
            QSlider::groove:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(31, 41, 55, 230),
                    stop:1 rgba(55, 65, 81, 230));
                width: 8px;
                border-radius: 4px;
                border: 1px solid rgba(255, 23, 68, 102);
            }
            QSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 23, 68, 204),
                    stop:1 rgba(255, 69, 0, 204));
                border: 2px solid rgba(255, 23, 68, 153);
                width: 20px;
                height: 15px;
                border-radius: 8px;
                margin: -6px;
            }
            QSlider::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 69, 0, 255),
                    stop:1 rgba(255, 23, 68, 255));
                border: 2px solid rgba(255, 69, 0, 255);
            }
            QSlider::sub-page:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 23, 68, 153),
                    stop:1 rgba(255, 69, 0, 153));
                border-radius: 4px;
            }
            QSlider::add-page:vertical {
                background: rgba(31, 41, 55, 128);
                border-radius: 4px;
            }
        """)
        self.density_slider.valueChanged.connect(self._on_density_changed)
        self.density_slider.move(self.width() - (button_spacing * 3) - 40, button_y + 50)
        self.density_slider.raise_()
        
        # Density label
        self.density_label = QLabel("Density", self)
        self.density_label.setStyleSheet("""
            QLabel {
                color: #FF1744;
                font-family: 'Arial Black', Arial, sans-serif;
                font-size: 10px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.density_label.setFixedSize(50, 20)
        self.density_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.density_label.move(self.width() - (button_spacing * 3) - 50, button_y + 210)
        self.density_label.raise_()
        
        # Lava toggle button
        self.particle_toggle = QPushButton("🌋", self)
        self.particle_toggle.setToolTip("Toggle Lava Drip Effects")
        self.particle_toggle.setFixedSize(40, 40)
        self.particle_toggle.setStyleSheet("""
            QPushButton {
                background-color: rgba(31, 41, 55, 230);
                border: 2px solid rgba(255, 23, 68, 204);
                border-radius: 20px;
                color: #FF1744;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Arial Black', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(255, 23, 68, 76);
                border: 2px solid rgba(255, 69, 0, 255);
                margin-top: -1px;
                margin-bottom: 1px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 23, 68, 153);
                margin-top: 1px;
                margin-bottom: -1px;
            }
        """)
        self.particle_toggle.clicked.connect(self._toggle_particles)
        self.particle_toggle.move(self.width() - button_spacing, button_y)
        self.particle_toggle.raise_()
        
        # Drain toggle button
        self.drain_toggle = QPushButton("🌊", self)
        self.drain_toggle.setToolTip("Toggle Pool Drainage")
        self.drain_toggle.setFixedSize(40, 40)
        self.drain_toggle.setStyleSheet("""
            QPushButton {
                background-color: rgba(31, 41, 55, 230);
                border: 2px solid rgba(0, 150, 255, 204);
                border-radius: 20px;
                color: #0096FF;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Arial Black', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(0, 150, 255, 76);
                border: 2px solid rgba(0, 200, 255, 255);
                margin-top: -1px;
                margin-bottom: 1px;
            }
            QPushButton:pressed {
                background-color: rgba(0, 150, 255, 153);
                margin-top: 1px;
                margin-bottom: -1px;
            }
        """)
        self.drain_toggle.clicked.connect(self._toggle_drain)
        self.drain_toggle.move(self.width() - (button_spacing * 2), button_y)
        self.drain_toggle.raise_()
        
        # Set enhanced dark theme
        self.setup_enhanced_dark_theme()
        
        logger.info("RagchefUI initialization complete")


    def _on_density_changed(self, value):
        """Handle particle density slider change."""
        density = value / 100.0  # Convert from 0-200 to 0.0-2.0
        if hasattr(self, 'particles') and self.particles:
            self.particles.set_particle_density(density)
            
        # Update tooltip with current value
        self.density_slider.setToolTip(f"Particle Density: {density:.1f}\n(Low ← → High)")

    def _toggle_particles(self):
        """Toggle lava drip effects."""
        if hasattr(self, 'particles') and self.particles:
            self.particles.toggle_particles()
            if self.particles.is_enabled:
                self.particle_toggle.setText("🌋")
                self.particle_toggle.setToolTip("Disable Lava Drip Effects")
            else:
                self.particle_toggle.setText("❄️")
                self.particle_toggle.setToolTip("Enable Lava Drip Effects")

    def _toggle_drain(self):
        """Toggle lava pool drainage."""
        if hasattr(self, 'particles') and self.particles:
            self.particles.toggle_drain()
            if self.particles.is_draining:
                self.drain_toggle.setText("🔧")
                self.drain_toggle.setToolTip("Stop Pool Drainage")
                self.drain_toggle.setStyleSheet(self.drain_toggle.styleSheet().replace("rgba(0, 150, 255", "rgba(255, 150, 0"))
            else:
                self.drain_toggle.setText("🌊")
                self.drain_toggle.setToolTip("Start Pool Drainage")
                self.drain_toggle.setStyleSheet(self.drain_toggle.styleSheet().replace("rgba(255, 150, 0", "rgba(0, 150, 255"))

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        
        # Update particle system
        if hasattr(self, 'particles') and self.particles:
            self.particles.setGeometry(0, 0, self.width(), self.height())
            self.particles.raise_()
        
        # Update control positions
        button_y = 10
        button_spacing = 50
        
        if hasattr(self, 'density_slider') and self.density_slider:
            self.density_slider.move(self.width() - (button_spacing * 3) - 40, button_y + 50)
            self.density_slider.raise_()
        
        if hasattr(self, 'density_label') and self.density_label:
            self.density_label.move(self.width() - (button_spacing * 3) - 50, button_y + 210)
            self.density_label.raise_()
        
        if hasattr(self, 'particle_toggle') and self.particle_toggle:
            self.particle_toggle.move(self.width() - button_spacing, button_y)
            self.particle_toggle.raise_()
        
        if hasattr(self, 'drain_toggle') and self.drain_toggle:
            self.drain_toggle.move(self.width() - (button_spacing * 2), button_y)
            self.drain_toggle.raise_()

    def setup_enhanced_dark_theme(self):
        """Enhanced dark theme compatible with PyQt6."""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(17, 24, 39, 245), 
                    stop:0.5 rgba(31, 41, 55, 235),
                    stop:1 rgba(17, 24, 39, 245));
                border: 2px solid rgba(255, 23, 68, 76);
                border-radius: 12px;
            }
            QWidget {
                color: #FFFFFF;
                font-family: 'Arial Black', Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(75, 85, 99, 230),
                    stop:0.5 rgba(55, 65, 81, 217),
                    stop:1 rgba(31, 41, 55, 230));
                border: 2px solid rgba(255, 23, 68, 102);
                border-radius: 8px;
                padding: 12px 20px;
                color: #FFFFFF;
                font-family: 'Arial Black', Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 23, 68, 76),
                    stop:0.5 rgba(75, 85, 99, 230),
                    stop:1 rgba(255, 23, 68, 76));
                border: 2px solid rgba(255, 23, 68, 204);
                margin-top: -2px;
                margin-bottom: 2px;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 23, 68, 153),
                    stop:1 rgba(31, 41, 55, 242));
                margin-top: 1px;
                margin-bottom: -1px;
            }
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(31, 41, 55, 217),
                    stop:0.5 rgba(55, 65, 81, 204),
                    stop:1 rgba(31, 41, 55, 217));
                border: 2px solid rgba(255, 23, 68, 127);
                border-radius: 12px;
                padding-top: 20px;
                margin-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #FF1744;
                font-family: 'Arial Black', Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
                subcontrol-position: top center;
                padding: 0 8px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 65, 81, 230),
                    stop:1 rgba(31, 41, 55, 242));
                border: 2px solid rgba(255, 23, 68, 76);
                border-radius: 8px;
                padding: 8px;
                color: #FFFFFF;
                font-family: 'Arial', sans-serif;
                font-size: 13px;
                font-weight: bold;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid rgba(255, 23, 68, 204);
            }
            QProgressBar {
                border: 2px solid rgba(255, 23, 68, 127);
                border-radius: 8px;
                text-align: center;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(31, 41, 55, 204),
                    stop:1 rgba(55, 65, 81, 204));
                font-family: 'Arial Black', Arial, sans-serif;
                font-weight: bold;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 23, 68, 230),
                    stop:0.5 rgba(255, 69, 0, 230),
                    stop:1 rgba(255, 23, 68, 230));
                border-radius: 6px;
            }
            QTabWidget::pane {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(31, 41, 55, 217),
                    stop:1 rgba(55, 65, 81, 217));
                border: 2px solid rgba(255, 23, 68, 102);
                border-radius: 10px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 65, 81, 230),
                    stop:1 rgba(31, 41, 55, 230));
                color: #FFFFFF;
                padding: 12px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 2px solid rgba(255, 23, 68, 76);
                margin-right: 2px;
                font-family: 'Arial Black', Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 23, 68, 204),
                    stop:1 rgba(220, 20, 60, 204));
                border: 2px solid rgba(255, 23, 68, 204);
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 23, 68, 102),
                    stop:1 rgba(55, 65, 81, 230));
                border: 2px solid rgba(255, 23, 68, 153);
            }
            QScrollBar {
                background: rgba(31, 41, 55, 179);
                width: 14px;
                height: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 23, 68, 153),
                    stop:1 rgba(75, 85, 99, 204));
                border-radius: 6px;
                min-height: 30px;
                border: 1px solid rgba(255, 23, 68, 76);
            }
            QScrollBar::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 23, 68, 204),
                    stop:1 rgba(255, 69, 0, 204));
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: rgba(31, 41, 55, 102);
            }
            QLabel {
                color: #FFFFFF;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QSpinBox, QCheckBox {
                font-family: 'Arial', sans-serif;
                font-size: 13px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QSpinBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 65, 81, 230),
                    stop:1 rgba(31, 41, 55, 242));
                border: 2px solid rgba(255, 23, 68, 76);
                border-radius: 6px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid rgba(255, 23, 68, 127);
                border-radius: 4px;
                background: rgba(31, 41, 55, 204);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 23, 68, 204),
                    stop:1 rgba(255, 69, 0, 204));
                border: 2px solid rgba(255, 23, 68, 204);
            }
            QComboBox::drop-down {
                border: none;
                background: rgba(255, 23, 68, 102);
                border-radius: 4px;
            }
            QComboBox::down-arrow {
                border: 2px solid rgba(255, 23, 68, 153);
                width: 8px;
                height: 8px;
            }
        """)
    
    def _setup_ui(self):
        """Set up the main UI components."""
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.research_tab = QWidget()
        self.generate_tab = QWidget()
        self.process_tab = QWidget()
        self.analyze_tab = QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(self.research_tab, "Research")
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.process_tab, "Process")
        self.tabs.addTab(self.analyze_tab, "Analyze")
        
        # Set up each tab
        self._setup_research_tab()
        self._setup_generate_tab()
        self._setup_process_tab()
        self._setup_analyze_tab()
    
    def _setup_research_tab(self):
        """Set up the Research tab."""
        layout = QVBoxLayout()
        
        # Research form
        form_group = QGroupBox("Research Settings")
        form_layout = QFormLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItem("Loading models...")
        form_layout.addRow("Ollama Model:", self.model_combo)
        
        # Populate models asynchronously
        self._populate_model_list()
        
        # Topic input
        self.topic_input = QLineEdit()
        form_layout.addRow("Research Topic:", self.topic_input)
        
        # Max papers
        self.max_papers_spin = QSpinBox()
        self.max_papers_spin.setRange(1, 20)
        self.max_papers_spin.setValue(5)
        form_layout.addRow("Max Papers:", self.max_papers_spin)
        
        # Max search results
        self.max_search_spin = QSpinBox()
        self.max_search_spin.setRange(1, 50)
        self.max_search_spin.setValue(10)
        form_layout.addRow("Max Search Results:", self.max_search_spin)
        
        # GitHub options
        self.include_github_check = QCheckBox("Include GitHub Repositories")
        form_layout.addRow("", self.include_github_check)
        
        self.github_repos_input = QLineEdit()
        self.github_repos_input.setPlaceholderText("Repository URLs (comma-separated)")
        self.github_repos_input.setEnabled(False)
        form_layout.addRow("GitHub Repos:", self.github_repos_input)
        
        # Connect checkbox to enable/disable repo input
        self.include_github_check.stateChanged.connect(
            lambda state: self.github_repos_input.setEnabled(state == Qt.CheckState.Checked)
        )
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Research button
        self.research_button = QPushButton("Start Research")
        self.research_button.clicked.connect(self._on_research_clicked)
        layout.addWidget(self.research_button)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.research_progress = QProgressBar()
        self.research_progress.setRange(0, 0)  # Indeterminate
        self.research_progress.setVisible(False)
        progress_layout.addWidget(self.research_progress)
        
        self.research_log = QTextEdit()
        self.research_log.setReadOnly(True)
        progress_layout.addWidget(self.research_log)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, stretch=1)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.research_results = QTextEdit()
        self.research_results.setReadOnly(True)
        results_layout.addWidget(self.research_results)
        
        save_layout = QHBoxLayout()
        self.save_research_button = QPushButton("Save Results")
        self.save_research_button.clicked.connect(self._on_save_research_clicked)
        self.save_research_button.setEnabled(False)
        save_layout.addWidget(self.save_research_button)
        
        self.proceed_to_generate_button = QPushButton("Proceed to Generate")
        self.proceed_to_generate_button.clicked.connect(self._on_proceed_to_generate_clicked)
        self.proceed_to_generate_button.setEnabled(False)
        save_layout.addWidget(self.proceed_to_generate_button)
        
        results_layout.addLayout(save_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        # Add conversation viewer
        self.conversation_viewer = ConversationViewer()
        layout.addWidget(self.conversation_viewer, stretch=1)
        
        self.research_tab.setLayout(layout)
    
    def _populate_model_list(self):
        """Populate the model selection combo box with available Ollama models."""
        try:
            # Get list of models from Ollama
            models = ollama.list()
            
            # Clear and populate combo box
            self.model_combo.clear()
            
            # Handle different response formats
            if hasattr(models, 'models'):
                # New ListResponse format
                model_names = [model.model for model in models.models if hasattr(model, 'model')]
            elif isinstance(models, list):
                # List of Model objects format
                model_names = [model.model for model in models if hasattr(model, 'model')]
            elif isinstance(models, dict) and 'models' in models:
                # Old format
                model_names = [model['name'] for model in models['models'] if 'name' in model]
            else:
                model_names = ['llama2']  # Default fallback
                
            self.model_combo.addItems(model_names)
            
            # Set default model
            default_model = "llama2"
            index = self.model_combo.findText(default_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                
        except Exception as e:
            logger.error(f"Error populating model list: {str(e)}")
            self.model_combo.clear()
            self.model_combo.addItem("llama2")  # Fallback to default

    def _setup_generate_tab(self):
        """Set up the Generate tab."""
        layout = QVBoxLayout()
        
        # Generation settings
        settings_group = QGroupBox("Generation Settings")
        settings_layout = QFormLayout()
        
        # Number of turns
        self.turns_spin = QSpinBox()
        self.turns_spin.setRange(1, 10)
        self.turns_spin.setValue(3)
        settings_layout.addRow("Conversation Turns:", self.turns_spin)
        
        # Expansion factor
        self.expansion_spin = QSpinBox()
        self.expansion_spin.setRange(1, 10)
        self.expansion_spin.setValue(3)
        settings_layout.addRow("Expansion Factor:", self.expansion_spin)
        
        # Hedging level
        self.hedging_combo = QComboBox()
        self.hedging_combo.addItems(["confident", "balanced", "cautious"])
        self.hedging_combo.setCurrentText("balanced")
        settings_layout.addRow("Hedging Level:", self.hedging_combo)
        
        # Clean checkbox
        self.clean_check = QCheckBox("Clean Expanded Dataset")
        self.clean_check.setChecked(True)
        settings_layout.addRow("", self.clean_check)
        
        # Static fields settings
        static_group = QGroupBox("Static Fields")
        static_layout = QVBoxLayout()
        
        self.human_static_check = QCheckBox("Keep Human Messages Static")
        self.human_static_check.setChecked(True)
        static_layout.addWidget(self.human_static_check)
        
        self.gpt_static_check = QCheckBox("Keep GPT Messages Static")
        self.gpt_static_check.setChecked(False)
        static_layout.addWidget(self.gpt_static_check)
        
        static_group.setLayout(static_layout)
        settings_layout.addRow("", static_group)
        
        # Output format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jsonl", "parquet", "csv", "all"])
        settings_layout.addRow("Output Format:", self.format_combo)
        
        # Output directory
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(str(self.research_manager.datasets_dir))
        self.output_dir_layout.addWidget(self.output_dir_input)
        
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self._on_browse_output_clicked)
        self.output_dir_layout.addWidget(self.browse_output_button)
        
        settings_layout.addRow("Output Directory:", self.output_dir_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Dataset")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.generate_button)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.generate_progress = QProgressBar()
        self.generate_progress.setRange(0, 0)  # Indeterminate
        self.generate_progress.setVisible(False)
        progress_layout.addWidget(self.generate_progress)
        
        self.generate_log = QTextEdit()
        self.generate_log.setReadOnly(True)
        progress_layout.addWidget(self.generate_log)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, stretch=1)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.generate_results = QTextEdit()
        self.generate_results.setReadOnly(True)
        results_layout.addWidget(self.generate_results)
        
        open_layout = QHBoxLayout()
        self.open_output_button = QPushButton("Open Output Directory")
        self.open_output_button.clicked.connect(self._on_open_output_clicked)
        open_layout.addWidget(self.open_output_button)
        
        self.proceed_to_analyze_button = QPushButton("Analyze Dataset")
        self.proceed_to_analyze_button.clicked.connect(self._on_proceed_to_analyze_clicked)
        self.proceed_to_analyze_button.setEnabled(False)
        open_layout.addWidget(self.proceed_to_analyze_button)
        
        results_layout.addLayout(open_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        self.generate_tab.setLayout(layout)
    
    def _setup_process_tab(self):
        """Set up the Process tab for existing papers."""
        layout = QVBoxLayout()
        
        # Input selection
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()
        
        self.input_dir_layout = QHBoxLayout()
        self.input_dir_label = QLabel("Input Directory:")
        self.input_dir_layout.addWidget(self.input_dir_label)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_layout.addWidget(self.input_dir_edit)
        
        self.browse_input_button = QPushButton("Browse...")
        self.browse_input_button.clicked.connect(self._on_browse_input_clicked)
        self.input_dir_layout.addWidget(self.browse_input_button)
        
        input_layout.addLayout(self.input_dir_layout)
        
        # Files list
        self.files_label = QLabel("Selected Files:")
        input_layout.addWidget(self.files_label)
        
        self.files_list = QTextEdit()
        self.files_list.setReadOnly(True)
        self.files_list.setMaximumHeight(100)
        input_layout.addWidget(self.files_list)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Processing settings
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QFormLayout()
        
        # Number of turns
        self.proc_turns_spin = QSpinBox()
        self.proc_turns_spin.setRange(1, 10)
        self.proc_turns_spin.setValue(3)
        settings_layout.addRow("Conversation Turns:", self.proc_turns_spin)
        
        # Expansion factor
        self.proc_expansion_spin = QSpinBox()
        self.proc_expansion_spin.setRange(1, 10)
        self.proc_expansion_spin.setValue(3)
        settings_layout.addRow("Expansion Factor:", self.proc_expansion_spin)
        
        # Clean checkbox
        self.proc_clean_check = QCheckBox("Clean Expanded Dataset")
        self.proc_clean_check.setChecked(True)
        settings_layout.addRow("", self.proc_clean_check)
        
        # Output format
        self.proc_format_combo = QComboBox()
        self.proc_format_combo.addItems(["jsonl", "parquet", "csv", "all"])
        settings_layout.addRow("Output Format:", self.proc_format_combo)
        
        # Output directory
        self.proc_output_dir_layout = QHBoxLayout()
        self.proc_output_dir_input = QLineEdit()
        self.proc_output_dir_input.setText(str(self.research_manager.datasets_dir))
        self.proc_output_dir_layout.addWidget(self.proc_output_dir_input)
        
        self.proc_browse_output_button = QPushButton("Browse...")
        self.proc_browse_output_button.clicked.connect(self._on_proc_browse_output_clicked)
        self.proc_output_dir_layout.addWidget(self.proc_browse_output_button)
        
        settings_layout.addRow("Output Directory:", self.proc_output_dir_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Process button
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self._on_process_clicked)
        layout.addWidget(self.process_button)
        
        # Progress area
        proc_progress_group = QGroupBox("Progress")
        proc_progress_layout = QVBoxLayout()
        
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 0)  # Indeterminate
        self.process_progress.setVisible(False)
        proc_progress_layout.addWidget(self.process_progress)
        
        self.process_log = QTextEdit()
        self.process_log.setReadOnly(True)
        proc_progress_layout.addWidget(self.process_log)
        
        proc_progress_group.setLayout(proc_progress_layout)
        layout.addWidget(proc_progress_group, stretch=1)
        
        # Results area
        proc_results_group = QGroupBox("Results")
        proc_results_layout = QVBoxLayout()
        
        self.process_results = QTextEdit()
        self.process_results.setReadOnly(True)
        proc_results_layout.addWidget(self.process_results)
        
        proc_results_group.setLayout(proc_results_layout)
        layout.addWidget(proc_results_group, stretch=1)
        
        self.process_tab.setLayout(layout)

    def _setup_analyze_tab(self):
        """Set up the Analyze tab for dataset analysis."""
        layout = QVBoxLayout()
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QFormLayout()
        
        # Original dataset
        self.orig_dataset_layout = QHBoxLayout()
        self.orig_dataset_input = QLineEdit()
        self.orig_dataset_layout.addWidget(self.orig_dataset_input)
        
        self.browse_orig_button = QPushButton("Browse...")
        self.browse_orig_button.clicked.connect(self._on_browse_orig_clicked)
        self.orig_dataset_layout.addWidget(self.browse_orig_button)
        
        dataset_layout.addRow("Original Dataset:", self.orig_dataset_layout)
        
        # Expanded dataset
        self.exp_dataset_layout = QHBoxLayout()
        self.exp_dataset_input = QLineEdit()
        self.exp_dataset_layout.addWidget(self.exp_dataset_input)
        
        self.browse_exp_button = QPushButton("Browse...")
        self.browse_exp_button.clicked.connect(self._on_browse_exp_clicked)
        self.exp_dataset_layout.addWidget(self.browse_exp_button)
        
        dataset_layout.addRow("Expanded Dataset:", self.exp_dataset_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout()
        
        self.basic_stats_check = QCheckBox("Basic Statistics")
        self.basic_stats_check.setChecked(True)
        analysis_layout.addWidget(self.basic_stats_check)
        
        self.quality_check = QCheckBox("Quality Analysis")
        self.quality_check.setChecked(True)
        analysis_layout.addWidget(self.quality_check)
        
        self.comparison_check = QCheckBox("Dataset Comparison")
        self.comparison_check.setChecked(True)
        analysis_layout.addWidget(self.comparison_check)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze Datasets")
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self.analyze_button)
        
        # Results area
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        results_layout.addWidget(self.analysis_results)
        
        save_analysis_button = QPushButton("Save Analysis Results")
        save_analysis_button.clicked.connect(self._on_save_analysis_clicked)
        results_layout.addWidget(save_analysis_button)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        self.analyze_tab.setLayout(layout)

    # Event handlers
    def _on_research_clicked(self):
        """Handle research button click."""
        logger.info("Starting research operation...")
        topic = self.topic_input.text().strip()
        if not topic:
            QMessageBox.warning(self, "Input Required", "Please enter a research topic.")
            return
            
        self.research_button.setEnabled(False)
        self.research_progress.setVisible(True)
        self.research_log.clear()
        
        # Get GitHub repos if enabled
        github_repos = None
        if self.include_github_check.isChecked():
            repos = self.github_repos_input.text().strip()
            if repos:
                github_repos = [r.strip() for r in repos.split(',')]
        
        # Get selected model
        model = self.model_combo.currentText()
        
        # Start research in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.research_topic,
            topic=topic,
            max_papers=self.max_papers_spin.value(),
            max_search_results=self.max_search_spin.value(),
            include_github=self.include_github_check.isChecked(),
            github_repos=github_repos,
            model_name=model  # Pass model name to research_topic
        )
        
        self.worker_thread.update_signal.connect(self._on_research_update)
        self.worker_thread.result_signal.connect(self._on_research_complete)
        self.worker_thread.error_signal.connect(self._on_research_error)
        
        self.worker_thread.start()
    
    def _on_research_update(self, message):
        """Handle research progress updates."""
        self.research_log.append(message)
        self.research_log.moveCursor(QTextCursor.MoveOperation.End)
        
        # Check if message contains conversation data
        if "human:" in message.lower() or "gpt:" in message.lower():
            # Extract role and content
            parts = message.split(":", 1)
            if len(parts) == 2:
                role = parts[0].strip().lower()
                content = parts[1].strip()
                self.conversation_viewer.add_message(role, content)
    
    def _on_research_complete(self, result):
        """Handle research completion."""
        self.research_button.setEnabled(True)
        self.research_progress.setVisible(False)
        self.save_research_button.setEnabled(True)
        self.proceed_to_generate_button.setEnabled(True)
        
        # Display summary
        summary = result.get('summary', 'No summary available')
        self.research_results.setPlainText(summary)
        
        # Store results for later use
        self.current_research_results = result
    
    def _on_research_error(self, error_msg):
        """Handle research errors."""
        logger.error(f"Research error: {error_msg}")
        self.research_button.setEnabled(True)
        self.research_progress.setVisible(False)
        
        # Create message box with error details but continue UI operation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Research Error")
        msg.setText("The research operation encountered some issues.")
        msg.setDetailedText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # Still show partial results if available
        if hasattr(self, 'current_research_results'):
            self.research_results.setPlainText(self.current_research_results.get('summary', 'Partial results available'))
            self.save_research_button.setEnabled(True)

    def _on_save_research_clicked(self):
        """Handle save research button click."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Research Results",
            "",
            "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if hasattr(self, 'current_research_results'):
                        json.dump(self.current_research_results, f, indent=2)
                    else:
                        f.write(self.research_results.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")

    def _on_proceed_to_generate_clicked(self):
        """Handle proceed to generate button click."""
        self.tabs.setCurrentIndex(1)  # Switch to Generate tab

    def _on_browse_output_clicked(self):
        """Handle browse output directory button click."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.research_manager.datasets_dir)
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def _on_browse_input_clicked(self):
        """Handle browse input directory button click."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            str(self.research_manager.papers_dir)
        )
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            # Update files list
            try:
                files = list(Path(dir_path).glob('*.txt')) + list(Path(dir_path).glob('*.md'))
                self.files_list.setPlainText('\n'.join(str(f) for f in files))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error reading directory: {str(e)}")

    def _on_proc_browse_output_clicked(self):
        """Handle browse output directory button click in process tab."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.research_manager.datasets_dir)
        )
        if dir_path:
            self.proc_output_dir_input.setText(dir_path)

    def _on_browse_orig_clicked(self):
        """Handle browse original dataset button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Original Dataset",
            str(self.research_manager.datasets_dir),
            "Dataset Files (*.jsonl *.parquet *.csv);;All Files (*.*)"
        )
        if file_path:
            self.orig_dataset_input.setText(file_path)

    def _on_browse_exp_clicked(self):
        """Handle browse expanded dataset button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Expanded Dataset",
            str(self.research_manager.datasets_dir),
            "Dataset Files (*.jsonl *.parquet *.csv);;All Files (*.*)"
        )
        if file_path:
            self.exp_dataset_input.setText(file_path)

    def _on_generate_clicked(self):
        """Handle generate dataset button click."""
        if not hasattr(self, 'current_research_results'):
            QMessageBox.warning(self, "Input Required", "Please complete research first or load existing papers.")
            return

        self.generate_button.setEnabled(False)
        self.generate_progress.setVisible(True)
        self.generate_log.clear()

        # Start generation in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.generate_conversation_dataset,
            papers=self.current_research_results.get('processed_papers', []),
            num_turns=self.turns_spin.value(),
            expansion_factor=self.expansion_spin.value(),
            clean=self.clean_check.isChecked()
        )

        self.worker_thread.update_signal.connect(self._on_generate_update)
        self.worker_thread.result_signal.connect(self._on_generate_complete)
        self.worker_thread.error_signal.connect(self._on_generate_error)

        self.worker_thread.start()

    def _on_generate_update(self, message):
        """Handle generation progress updates."""
        self.generate_log.append(message)
        self.generate_log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_generate_complete(self, result):
        """Handle generation completion."""
        self.generate_button.setEnabled(True)
        self.generate_progress.setVisible(False)
        self.proceed_to_analyze_button.setEnabled(True)

        # Show summary in results
        summary = f"Generated {len(result.get('conversations', []))} conversations\n"
        summary += f"Expanded to {len(result.get('expanded_conversations', []))} variations\n"
        summary += f"Output saved to: {result.get('output_path', 'unknown')}"
        self.generate_results.setPlainText(summary)

    def _on_generate_error(self, error_msg):
        """Handle generation errors."""
        logger.error(f"Generation error: {error_msg}")
        self.generate_button.setEnabled(True)
        self.generate_progress.setVisible(False)
        QMessageBox.critical(self, "Generation Error", error_msg)

    def _on_open_output_clicked(self):
        """Handle open output directory button click."""
        output_dir = self.output_dir_input.text()
        if not output_dir:
            output_dir = str(self.research_manager.datasets_dir)
        
        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            else:
                webbrowser.open(f'file://{output_dir}')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")

    def _on_proceed_to_analyze_clicked(self):
        """Handle proceed to analyze button click."""
        self.tabs.setCurrentIndex(3)  # Switch to Analyze tab

    def _on_process_clicked(self):
        """Handle process files button click."""
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Input Required", "Please select an input directory.")
            return

        self.process_button.setEnabled(False)
        self.process_progress.setVisible(True)
        self.process_log.clear()

        # Get list of paper files
        paper_files = list(Path(input_dir).glob('*.txt')) + list(Path(input_dir).glob('*.md'))
        if not paper_files:
            QMessageBox.warning(self, "No Files Found", "No text or markdown files found in the input directory.")
            self.process_button.setEnabled(True)
            self.process_progress.setVisible(False)
            return

        # Start processing in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.process_paper_files,
            paper_files=paper_files,
            output_format=self.proc_format_combo.currentText(),
            num_turns=self.proc_turns_spin.value(),
            expansion_factor=self.proc_expansion_spin.value(),
            clean=self.proc_clean_check.isChecked()
        )

        self.worker_thread.update_signal.connect(self._on_process_update)
        self.worker_thread.result_signal.connect(self._on_process_complete)
        self.worker_thread.error_signal.connect(self._on_process_error)

        self.worker_thread.start()

    def _on_process_update(self, message):
        """Handle process progress updates."""
        self.process_log.append(message)
        self.process_log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_process_complete(self, result):
        """Handle process completion."""
        self.process_button.setEnabled(True)
        self.process_progress.setVisible(False)

        # Show summary in results
        summary = f"Processed {result.get('conversations_count', 0)} conversations\n\nOutput files:"
        for fmt, path in result.get('output_paths', {}).items():
            if isinstance(path, str):
                summary += f"\n{fmt}: {path}"
        self.process_results.setPlainText(summary)

    def _on_process_error(self, error_msg):
        """Handle process errors."""
        self.process_button.setEnabled(True)
        self.process_progress.setVisible(False)
        QMessageBox.critical(self, "Process Error", error_msg)

    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        orig_path = self.orig_dataset_input.text().strip()
        exp_path = self.exp_dataset_input.text().strip()
        
        if not orig_path or not exp_path:
            QMessageBox.warning(self, "Input Required", "Please select both original and expanded datasets.")
            return
        
        self.analyze_button.setEnabled(False)
        self.analysis_results.clear()
        
        # Create a wrapper function that uses the dataset expander's analyze method
        async def analyze_datasets_wrapper(orig_path, exp_path, analysis_options, callback=None):
            """Wrapper function to analyze datasets using the dataset expander."""
            try:
                import json
                from pathlib import Path
                
                if callback:
                    callback("Loading datasets...")
                
                # Load datasets based on file format
                def load_dataset(file_path):
                    path = Path(file_path)
                    if path.suffix == '.jsonl':
                        conversations = []
                        with open(path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    conversations.append(json.loads(line))
                        return conversations
                    elif path.suffix == '.json':
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Handle different JSON formats
                            if isinstance(data, list):
                                return data
                            elif isinstance(data, dict) and 'conversations' in data:
                                return data['conversations']
                            else:
                                return [data]  # Assume single conversation
                    else:
                        raise ValueError(f"Unsupported file format: {path.suffix}")
                
                if callback:
                    callback("Loading original dataset...")
                orig_conversations = load_dataset(orig_path)
                
                if callback:
                    callback("Loading expanded dataset...")
                exp_conversations = load_dataset(exp_path)
                
                if callback:
                    callback("Analyzing datasets...")
                
                # Use the dataset expander's analyze_expanded_dataset method
                analysis_results = self.research_manager.dataset_expander.analyze_expanded_dataset(
                    orig_conversations, 
                    exp_conversations
                )
                
                if callback:
                    callback("Analysis complete!")
                
                return analysis_results
                
            except Exception as e:
                if callback:
                    callback(f"Error during analysis: {str(e)}")
                raise
        
        # Start analysis in worker thread
        self.worker_thread = WorkerThread(
            analyze_datasets_wrapper,
            orig_path=orig_path,
            exp_path=exp_path,
            analysis_options={
                "basic_stats": self.basic_stats_check.isChecked(),
                "quality": self.quality_check.isChecked(),
                "comparison": self.comparison_check.isChecked()
            }
        )
        
        self.worker_thread.update_signal.connect(self._append_to_analysis_results)
        self.worker_thread.result_signal.connect(self._on_analysis_complete)
        self.worker_thread.error_signal.connect(self._on_analysis_error)
        
        self.worker_thread.start()
    
    def _append_to_analysis_results(self, text):
        """Append text to analysis results."""
        self.analysis_results.append(text)
        self.analysis_results.moveCursor(QTextCursor.MoveOperation.End)
    
    def _on_analysis_complete(self, results):
        """Handle analysis completion."""
        self.analyze_button.setEnabled(True)
        
        # Format and display results
        formatted_results = json.dumps(results, indent=2)
        self.analysis_results.setPlainText(formatted_results)
    
    def _on_analysis_error(self, error_msg):
        """Handle analysis errors."""
        logger.error(f"Analysis error: {error_msg}")
        self.analyze_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_msg)
    
    def _on_save_analysis_clicked(self):
        """Save analysis results to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Results",
            "",
            "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.analysis_results.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")

    def paintEvent(self, event):
        """Paint the quantum fluid mesh system."""
        # This is correct - the main window doesn't need custom painting
        # The QuantumFluidSystem widget handles all the visual effects
        super().paintEvent(event)