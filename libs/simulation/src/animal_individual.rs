use lib_genetic_algorithm::Chromosome;
use crate::*;

pub struct AnimalIndividual {
    fitness: f32,
    chromosome: Chromosome,
}

impl ga::Individual for AnimalIndividual {
    fn fitness(&self) -> f32 {
        self.fitness
    }

    fn chromosome(&self) -> &Chromosome {
        &self.chromosome
    }

    fn create(chromosome: Chromosome) -> Self {
        Self {
            fitness: 0.0,
            chromosome,
        }
    }
}

impl AnimalIndividual {
    pub fn from_animal(animal: &Animal) -> Self {
        Self {
            fitness: animal.satiation as f32,
            chromosome: animal.as_chromosome(),
        }
    }

    pub fn into_animal(self, rng: &mut dyn rand::RngCore) -> Animal {
        Animal::from_chromosome(self.chromosome, rng)
    }
}