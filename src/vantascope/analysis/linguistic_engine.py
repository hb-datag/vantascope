"""
Linguistic Output Engine - Convert model predictions to scientific text.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..utils.logging import logger


@dataclass
class StructureAnalysis:
    """Complete structure analysis with linguistic descriptions."""
    
    # Quantitative values
    energy: float
    energy_uncertainty: float
    crystallinity: float
    defect_density: float
    stability_score: float
    num_atoms: int
    defect_ratio: float
    
    # Classifications
    primary_defect_type: str
    defect_probabilities: Dict[str, float]
    quality_grade: str
    
    # Linguistic descriptions
    summary: str
    geometric_description: str
    topological_description: str
    recommendations: str


class LinguisticEngine:
    """Generate scientific linguistic output from model predictions."""
    
    DEFECT_TYPES = ['stone_wales', 'vacancy', 'grain_boundary', 'pristine']
    
    DEFECT_DESCRIPTIONS = {
        'stone_wales': 'Stone-Wales defect (90° bond rotation creating 5-7 ring pairs)',
        'vacancy': 'Vacancy defect (missing carbon atom)',
        'grain_boundary': 'Grain boundary (interface between crystalline domains)',
        'pristine': 'Pristine graphene (perfect hexagonal lattice)'
    }
    
    PROPERTY_IMPLICATIONS = {
        'stone_wales': {
            'electronic': 'Local bandgap opening of ~0.1-0.3 eV',
            'mechanical': 'Slight reduction in fracture strength (~5-10%)',
            'chemical': 'Enhanced reactivity at defect site'
        },
        'vacancy': {
            'electronic': 'Strong localized states, potential magnetic moments',
            'mechanical': 'Significant stress concentration, crack initiation site',
            'chemical': 'Highly reactive dangling bonds'
        },
        'grain_boundary': {
            'electronic': 'Carrier scattering, reduced mobility',
            'mechanical': 'Potential weak point under tensile stress',
            'chemical': 'Enhanced chemical reactivity along boundary'
        },
        'pristine': {
            'electronic': 'Zero bandgap, high carrier mobility (~200,000 cm²/V·s)',
            'mechanical': 'Maximum theoretical strength (~130 GPa)',
            'chemical': 'Chemically inert basal plane'
        }
    }
    
    def __init__(self):
        logger.info("📝 Linguistic Engine initialized")
    
    def analyze(self, 
                model_outputs: Dict[str, torch.Tensor],
                metadata: Dict) -> StructureAnalysis:
        """Generate complete linguistic analysis from model outputs."""
        
        # Extract predictions
        energy_mean = model_outputs['energy_mean'].item()
        energy_std = model_outputs['energy_std'].item()
        
        properties = model_outputs['properties']
        crystallinity = properties['crystallinity'].item()
        defect_density = properties['defect_density'].item()
        stability = properties['stability'].item()
        defect_probs = properties['defect_probs'].squeeze().tolist()
        
        # Get metadata
        num_atoms = metadata.get('num_atoms', 0)
        defect_ratio = metadata.get('defect_ratio', 0.0)
        
        # Determine primary defect type
        defect_prob_dict = {
            name: prob for name, prob in zip(self.DEFECT_TYPES, defect_probs)
        }
        primary_defect = max(defect_prob_dict, key=defect_prob_dict.get)
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(crystallinity, defect_density)
        
        # Generate descriptions
        summary = self._generate_summary(
            energy_mean, energy_std, crystallinity, defect_density,
            primary_defect, defect_prob_dict[primary_defect], num_atoms
        )
        
        geometric_desc = self._generate_geometric_description(
            crystallinity, num_atoms, metadata
        )
        
        topological_desc = self._generate_topological_description(
            primary_defect, defect_prob_dict, defect_density, defect_ratio
        )
        
        recommendations = self._generate_recommendations(
            primary_defect, crystallinity, stability, defect_density
        )
        
        return StructureAnalysis(
            energy=energy_mean,
            energy_uncertainty=energy_std,
            crystallinity=crystallinity,
            defect_density=defect_density,
            stability_score=stability,
            num_atoms=num_atoms,
            defect_ratio=defect_ratio,
            primary_defect_type=primary_defect,
            defect_probabilities=defect_prob_dict,
            quality_grade=quality_grade,
            summary=summary,
            geometric_description=geometric_desc,
            topological_description=topological_desc,
            recommendations=recommendations
        )
    
    def _determine_quality_grade(self, crystallinity: float, defect_density: float) -> str: 
        """Determine overall quality grade."""
        score = crystallinity * 0.6 + (1 - defect_density) * 0.4
        
        if score > 0.9:
            return "A+ (Exceptional)"
        elif score > 0.8:
            return "A (Excellent)"
        elif score > 0.7:
            return "B (Good)"
        elif score > 0.6:
            return "C (Acceptable)"
        elif score > 0.5:
            return "D (Poor)"
        else:
            return "F (Defective)"
    
    def _generate_summary(self, energy: float, energy_std: float,
                         crystallinity: float, defect_density: float,
                         primary_defect: str, defect_prob: float,
                         num_atoms: int) -> str:
        """Generate executive summary."""
        
        quality = "high" if crystallinity > 0.8 else "moderate" if crystallinity > 0.6 else "low"
        
        summary = f"""STRUCTURAL ANALYSIS SUMMARY

This graphene structure contains {num_atoms} carbon atoms with {quality} crystallinity ({crystallinity*100:.1f}%).

Total Energy: {energy:.1f} ± {energy_std:.1f} eV
Primary Classification: {self.DEFECT_DESCRIPTIONS[primary_defect]} (confidence: {defect_prob*100:.1f}%)
Defect Density: {defect_density*100:.2f}%

"""
        
        if primary_defect == 'pristine':
            summary += "This sample exhibits excellent structural order suitable for high-performance electronic applications."
        else:
            implications = self.PROPERTY_IMPLICATIONS[primary_defect]
            summary += f"Detected defects will affect material properties:\n"
            summary += f"  • Electronic: {implications['electronic']}\n"
            summary += f"  • Mechanical: {implications['mechanical']}\n"
            summary += f"  • Chemical: {implications['chemical']}"
        
        return summary
    
    def _generate_geometric_description(self, crystallinity: float, 
                                        num_atoms: int, metadata: Dict) -> str:
        """Generate description of geometric features."""
        
        # Calculate ideal atom count for pristine graphene
        # ~38.2 atoms/nm² for graphene, 3.5nm × 3.5nm ≈ 468 atoms ideal
        ideal_atoms = 468
        atom_deficit = ideal_atoms - num_atoms
        
        desc = f"""GEOMETRIC ANALYSIS

Lattice Parameters:
  • Unit cell: 3.5 nm × 3.5 nm (12.25 nm² area)
  • Atom count: {num_atoms} (ideal: ~{ideal_atoms})
  • Atom deficit: {atom_deficit} atoms ({abs(atom_deficit)/ideal_atoms*100:.1f}% deviation) 
  • Crystallinity index: {crystallinity*100:.1f}%

Lattice Quality Assessment:
"""
        
        if crystallinity > 0.9:
            desc += "  • Highly ordered hexagonal lattice with minimal distortions\n"
            desc += "  • Bond lengths within 1% of ideal (1.42 Å)\n"
            desc += "  • Angular distortion < 2° from ideal 120°"
        elif crystallinity > 0.7:
            desc += "  • Moderately ordered lattice with some local distortions\n"
            desc += "  • Bond length variations of 2-5%\n"
            desc += "  • Some angular distortions present (2-5°)"
        else:
            desc += "  • Significant lattice disorder detected\n"
            desc += "  • Bond length variations > 5%\n"
            desc += "  • Multiple angular distortions present"
        
        return desc
    
    def _generate_topological_description(self, primary_defect: str,
                                          defect_probs: Dict[str, float],
                                          defect_density: float,
                                          defect_ratio: float) -> str:
        """Generate description of topological features."""
        
        desc = f"""TOPOLOGICAL ANALYSIS

Defect Classification (probability):
"""
        
        # Sort by probability
        sorted_defects = sorted(defect_probs.items(), key=lambda x: x[1], reverse=True)
        
        for defect_type, prob in sorted_defects:
            indicator = "●" if prob > 0.5 else "○"
            desc += f"  {indicator} {defect_type.replace('_', ' ').title()}: {prob*100:.1f}%\n"
        
        desc += f"""
Defect Metrics:
  • Overall defect density: {defect_density*100:.2f}%
  • Atomic defect ratio: {defect_ratio*100:.2f}%
  • Primary defect type: {primary_defect.replace('_', ' ').title()}

Topological Features:
"""
        
        if primary_defect == 'pristine':
            desc += "  • No significant topological defects detected\n"
            desc += "  • Consistent six-fold coordination throughout\n"
            desc += "  • Uniform electron density distribution"
        elif primary_defect == 'stone_wales':
            desc += "  • Pentagon-heptagon (5-7) ring pairs detected\n"
            desc += "  • Local curvature induced by bond rotation\n"
            desc += "  • Preserved three-fold coordination"
        elif primary_defect == 'vacancy':
            desc += "  • Missing atom sites detected\n"
            desc += "  • Dangling bonds present at vacancy edges\n"
            desc += "  • Local structural reconstruction expected"
        elif primary_defect == 'grain_boundary':
            desc += "  • Interface between misoriented domains\n"
            desc += "  • Mixed pentagon-heptagon structures along boundary\n"
            desc += "  • Strain accumulation at boundary region"
        
        return desc
    
    def _generate_recommendations(self, primary_defect: str,
                                  crystallinity: float, stability: float,
                                  defect_density: float) -> str:
        """Generate application recommendations."""
        
        rec = """RECOMMENDATIONS

Suitable Applications:
"""
        
        if crystallinity > 0.85 and defect_density < 0.05:
            rec += """  ✓ High-frequency electronics (excellent carrier mobility)
  ✓ Transparent conductive films
  ✓ Mechanical reinforcement composites
  ✓ Thermal management applications
"""
        elif crystallinity > 0.7:
            rec += """  ✓ General electronic applications
  ✓ Sensor platforms (enhanced reactivity may be beneficial)
  ✓ Energy storage electrodes
  △ Limited for high-mobility applications
"""
        else:
            rec += """  ✓ Chemical functionalization substrates
  ✓ Catalysis support (high defect reactivity)
  ✗ Not recommended for electronic applications
  ✗ Not suitable for mechanical reinforcement
"""
        
        rec += "\nOptimization Suggestions:\n"
        
        if primary_defect == 'vacancy':
            rec += "  • Consider hydrogen passivation of dangling bonds\n"
            rec += "  • Thermal annealing may partially heal vacancies\n"
        elif primary_defect == 'stone_wales':
            rec += "  • High-temperature annealing (>1000°C) can reverse Stone-Wales defects\n"
            rec += "  • Defects may be beneficial for specific catalytic applications\n"
        elif primary_defect == 'grain_boundary':
            rec += "  • Consider CVD growth optimization for larger single crystals\n"
            rec += "  • Grain boundary engineering for specific transport properties\n"
        
        if stability < 0.7:
            rec += "  ⚠ Structure shows low stability - handle with care\n"
            rec += "  ⚠ May be susceptible to environmental degradation\n"
        
        return rec
    
    def format_full_report(self, analysis: StructureAnalysis) -> str:
        """Format complete analysis report."""
        
        divider = "=" * 60 + "\n"
        
        report = divider
        report += "           VANTASCOPE STRUCTURAL ANALYSIS REPORT\n"
        report += divider + "\n"
        
        report += f"Quality Grade: {analysis.quality_grade}\n\n"
        report += analysis.summary + "\n\n"
        report += divider
        report += analysis.geometric_description + "\n\n"
        report += divider
        report += analysis.topological_description + "\n\n"
        report += divider
        report += analysis.recommendations + "\n"
        report += divider
        
        return report


def create_linguistic_engine() -> LinguisticEngine:
    """Factory function."""
    return LinguisticEngine()
