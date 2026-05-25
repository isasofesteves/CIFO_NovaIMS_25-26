#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        IMPORTS                                                                           #
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import random
import time
import cupy as cp

#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        PARAMETERS                                                                        #
#------------------------------------------------------------------------------------------------------------------------------------------#

IMG_W, IMG_H    = 300, 400
NUM_TRIANGLES   = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        FUNCTIONS                                                                         #
#------------------------------------------------------------------------------------------------------------------------------------------#

def random_triangle():
    
    """
    Creates a random triangle represented as an array of 10 integers:
    - 6 integers for the coordinates of the 3 vertices (x1, y1, x2, y2, x3, y3)
    - 4 integers for the RGBA color (R, G, B, A)    
    """

    coords = [
        random.randint(0, IMG_W - 1),
        random.randint(0, IMG_H - 1),
        random.randint(0, IMG_W - 1),
        random.randint(0, IMG_H - 1),
        random.randint(0, IMG_W - 1),
        random.randint(0, IMG_H - 1),
    ]
    color = [
        random.randint(0, 255),    # R
        random.randint(0, 255),    # G
        random.randint(0, 255),    # B
        random.randint(20, 80),    # A
    ]
    return np.array(coords + color, dtype=np.int32)  # 10 values


def random_individual():

    """Return a random individual: array of 100 genes."""

    return np.array([random_triangle() for _ in range(NUM_TRIANGLES)], dtype=np.int32)


def decode(individual):
    
    """ 
    Returns list of (points, color) where
    points = [(x1,y1),(x2,y2),(x3,y3)]
    color  = (R, G, B, A)
    """

    triangles = []

    for g in individual:
        points = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5])]
        color  = (g[6], g[7], g[8], g[9])  # RGBA
        triangles.append((points, color))
        
    return triangles

# -----------------------------------------------------------------------------------------------


_render_kernel_opt = cp.RawKernel(r'''                               
// IMAGE AND POPULATION CONSTANTS

#define IMG_W 300
#define IMG_H 400
#define NUM_TRIANGLES 100
                                  

// THREAD ASSIGNMENT 
extern "C" __global__
void render_triangles_cuda(
    float* output,
    const float* triangles,
    const float* bboxes,
    int pop_size
)
{   // each thread is responsible for a pixel (x,y) of an individual
    int x = blockIdx.x * blockDim.x + threadIdx.x; // thread x pixel coord
    int y = blockIdx.y * blockDim.y + threadIdx.y; // thread y pixel coord
    int indiv_idx  = blockIdx.z; // image id 

    if (x >= IMG_W || y >= IMG_H || indiv_idx >= pop_size) return; // stop outside image bounds

    const float cx = (float)x + 0.5f; // changing coords. to pixel centers
    const float cy = (float)y + 0.5f;

    float acc_r = 0.0f; // pixel starts completely see through
    float acc_g = 0.0f;
    float acc_b = 0.0f;
    float acc_a = 0.0f;

    const float* indiv_triangles = triangles + (size_t)indiv_idx * NUM_TRIANGLES * 10;
    const float* indiv_bboxes = bboxes + (size_t)indiv_idx * NUM_TRIANGLES * 4;
                                  
    // iterating over the individual's triangles
    for (int t = 0; t < NUM_TRIANGLES; ++t) {

        // bounding box: [minX, minY, maxX, maxY]
        const float minX = indiv_bboxes[t*4 + 0];
        const float minY = indiv_bboxes[t*4 + 1];
        const float maxX = indiv_bboxes[t*4 + 2];
        const float maxY = indiv_bboxes[t*4 + 3];

        // reject pixels outside of the bounding box
        if (cx < minX || cx > maxX || cy < minY || cy > maxY)
            continue;
                                  
        // triangle: [x0,y0, x1,y1, x2,y2, R,G,B,A]
        const float* triangle = indiv_triangles + t * 10;

        float x0 = triangle[0], y0 = triangle[1];
        float x1 = triangle[2], y1 = triangle[3];
        float x2 = triangle[4], y2 = triangle[5];

        // normalize alpha (A) and skip invisible triangles
        float alpha = fminf(fmaxf(triangle[9] / 255.0f, 0.0f), 1.0f);
        if (alpha <= 0.0f) continue;

        // skip very small triangles (area ~ 0)
        float area = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0); // can be negative
        
        float abs_area = fabsf(area); 
        if (abs_area < 0.2f) continue;

        // barycentric coordinates test
        float bc0 = ((y1 - y2)*(cx - x2) + (x2 - x1)*(cy - y2)) / abs_area;
        float bc1 = ((y2 - y0)*(cx - x2) + (x0 - x2)*(cy - y2)) / abs_area;
        float bc2 = 1.0f - bc0 - bc1;
                                  
        // if any of the coords. is negative, the pixel is outside the triangle
        if (bc0 < 0.0f || bc1 < 0.0f || bc2 < 0.0f) continue;

        // alpha composition
        float inv_alpha = 1.0f - alpha; // how much of the old canvas shows through

        // accumulating the colors of each triangle
        acc_r = triangle[6] * alpha + acc_r * inv_alpha;
        acc_g = triangle[7] * alpha + acc_g * inv_alpha;
        acc_b = triangle[8] * alpha + acc_b * inv_alpha;
        acc_a = 255.0f * alpha + acc_a * inv_alpha;
    }   
            
    // write colors into final output (pop_size, H, W, 4)
    int idx = ((indiv_idx * IMG_H + y) * IMG_W + x) * 4;

    output[idx + 0] = acc_r;
    output[idx + 1] = acc_g;
    output[idx + 2] = acc_b;
    output[idx + 3] = acc_a; }                       

''', 'render_triangles_cuda')


def compute_bboxes(population):

    # Creating a bounding box array: each traingle gets 4 box vertices
    bboxes = np.zeros(
        (population.shape[0], NUM_TRIANGLES, 4),
        dtype=np.float32)

    x = population[:, :, [0,2,4]] # population x coordinates
    y = population[:, :, [1,3,5]] # population y coordinates

    bboxes[:,:,0] = x.min(axis=2)  # minX
    bboxes[:,:,1] = y.min(axis=2)  # minY
    bboxes[:,:,2] = x.max(axis=2)  # maxX
    bboxes[:,:,3] = y.max(axis=2)  # maxY

    return bboxes

def render_population_cuda(population):

    population = np.asarray(population, dtype=np.float32)

    N = population.shape[0] # population size 

    bboxes = compute_bboxes(population) # getting the bounding box array for the population

    triangles_gpu = cp.asarray(population) # gpu transfer
    bboxes_gpu = cp.asarray(bboxes)

    output_gpu = cp.zeros((N, IMG_H, IMG_W, 4),dtype=cp.float32) # output canvas

    threads = (16, 16, 1) # 16x16 threads (256) per block

    blocks = (
        (IMG_W + threads[0] - 1) // threads[0], # floor division 
        (IMG_H + threads[1] - 1) // threads[1], 
        N)

    # running the renderer 
    _render_kernel_opt(
        blocks, # gridsize by blocks
        threads, # block size (16x16)
        (   # arguments of the render_triangles_cuda kernel function
            output_gpu.ravel(), # output image tensor
            triangles_gpu.ravel(), # triangle data
            bboxes_gpu.ravel(), # bounding boxes
            np.int32(N) # pop size
        )
    )

    return output_gpu


def render(individual):

    population = np.expand_dims(individual, axis=0)

    # Render using CUDA pipeline
    rendered = render_population_cuda(population)

    # Remove batch dimension
    img = cp.asnumpy(rendered[0])

    return img.astype(np.float32)



# -----------------------------------------------------------------------------------------------



#Fitness Function

def population_fitness_rmse(rendered_population, target):

    target_gpu = cp.asarray(target, dtype=cp.float32)

    diff = rendered_population - target_gpu[None, :, :, :]

    rmse = cp.sqrt(cp.mean(diff * diff, axis=(1,2,3)))

    return cp.asnumpy(rmse).tolist()



# Selection Function------------------------------------------------------------------------------
def tournament_selection(population, fitnesses, k):

    """Select an individual from the population using tournament selection."""

    selected = random.sample(list(zip(population, fitnesses)), k) # randomly select k individuals
    selected.sort(key=lambda x: x[1]) # sort by fitness (lower is better)

    return selected[0][0].copy() # return best individual's chromosome


# Mutation & Crossover Functions------------------------------------------------------------------
def _clamp(v, lo=0, hi=255):

    """ Convert the value to an integer and limit it to the valid RGB range [0, 255] """

    return max(lo, min(hi, int(v)))


def mixed_mutation(indiv, mut_prob, weights=None):
    """ Randomly selects one of several mutation operators to apply to the individual, based on specified weights.
    parameters:
        indiv: Individual to mutate.
        mut_prob: Mutation probability.
        weights: List of probabilities for each mutation operator (must sum to 1). If None, defaults to equal weights.
    returns:
        indiv: Mutated individual.

    """
    operators = [
        color_soft_mutation,
        creep_mutation,
        triangle_replacement_mutation,
        color_hard_mutation,
        triangle_sort_mutation,
        alpha_focus_mutation,
        shrink_triangle_mutation,
        grow_triangle_mutation,
        translate_triangle_mutation
    ]
    
    if weights is None:
        weights = [0.25, 0.20, 0.05, 0.05, 0.15, 0.10, 0.05, 0.10, 0.05]  # default

    chosen = random.choices(operators, weights=weights, k=1)[0]
    return chosen(indiv, mut_prob)


def color_soft_mutation(indiv, mut_prob, delta=30):

    """
    Mutates the color (RGBA) of a random triangle with a small offset.

    Each channel is perturbed by a uniform value in [-delta, +delta].
    The result is clamped to [0, 255], keeping the color close to the
    original

    Parameters:
        indiv:    Individual (array of triangles).
        mut_prob: Mutation probability.
        delta:    Maximum displacement per channel (default 30).

    Returns:
        indiv: Mutated individual.
    """

    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy()

        # Assuming color occupies the last 4 values of each triangle [r, g, b, a]
        for c in range(-4, 0):
            tri[c] = _clamp(tri[c] + random.randint(-delta, delta))

        mutated[idx] = tri

    return mutated


def color_hard_mutation(indiv, mut_prob):
    
    """
    Replaces the color (RGBA) of a random triangle with fully random values.

    Unlike soft mutation, there is no relation to the previous color 

    Parameters:
        indiv:    Individual (array of triangles).
        mut_prob: Mutation probability.

    Returns:
        indiv: Mutated individual.
    """

    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy()

        # Full replacement of RGBA channels
        for c in range(-4, 0):
            tri[c] = random.randint(0, 255)

        mutated[idx] = tri

    return mutated


def triangle_replacement_mutation(indiv, mut_prob):
    
    """
    Replaces an entire triangle (vertices + color) with a newly generated
    random one via random_triangle().

    Parameters:
        indiv:    Individual (array of triangles).
        mut_prob: Mutation probability.

    Returns:
        indiv: Mutated individual.
    """

    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        mutated[idx] = random_triangle()

    return mutated


def creep_mutation(indiv, mut_prob, vertex_delta=15, color_delta=20):
    
    """
    Slightly perturbs the vertices AND/OR color of a random triangle.

    Each component (x, y of each vertex, RGBA channels) is independently
    perturbed with probability 0.5, within +-delta.

    Parameters:
        indiv:        Individual (array of triangles).
        mut_prob:     Probability of applying the mutation.
        vertex_delta: Maximum displacement per coordinate.
        color_delta:  Maximum displacement per color channel.

    Returns:
        indiv: Mutated individual.
    """

    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy().astype(float)

        # Assuming structure [x1,y1, x2,y2, x3,y3, r,g,b,a]
        num_genes  = len(tri)
        num_color  = 4
        num_vertex = num_genes - num_color

        # Perturb vertices
        for i in range(num_vertex):
            if random.random() < 0.5:
                tri[i] += random.randint(-vertex_delta, vertex_delta)

        # Perturb color
        for i in range(num_vertex, num_genes):
            if random.random() < 0.5:
                tri[i] = _clamp(tri[i] + random.randint(-color_delta, color_delta))

        mutated[idx] = tri.astype(mutated.dtype)

    return mutated

def triangle_sort_mutation(indiv, mut_prob):
    """ Randomly selects two triangles and swaps their positions in the individual's array."""
    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        i, j = random.sample(range(NUM_TRIANGLES), 2)
        mutated[i], mutated[j] = mutated[j].copy(), mutated[i].copy()
    return mutated

def alpha_focus_mutation(indiv, mut_prob, delta=15):
    """ Specifically mutates the alpha channel of a random triangle to increase or decrease its opacity"""
    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy()
        tri[-1] = _clamp(tri[-1] + random.randint(-delta, delta))  # mutate alpha channel (last value)
        mutated[idx] = tri
    return mutated

def shrink_triangle_mutation(indiv, mut_prob, factor=0.5):
    """ Shrinks a random triangle towards its centroid by a given factor (0 < factor < 1) """
    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy().astype(float)
        cx = (tri[0] + tri[2] + tri[4]) / 3
        cy = (tri[1] + tri[3] + tri[5]) / 3
        for i in range(3):
            tri[2*i]   = _clamp(cx + factor * (tri[2*i]   - cx), 0, IMG_W-1)
            tri[2*i+1] = _clamp(cy + factor * (tri[2*i+1] - cy), 0, IMG_H-1)
        mutated[idx] = tri.astype(mutated.dtype)
    return mutated

def grow_triangle_mutation(indiv, mut_prob, factor=1.5):
    """Grows a random triangle away from its centroid by a given factor (> 1)"""
    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy().astype(float)
        cx = (tri[0] + tri[2] + tri[4]) / 3
        cy = (tri[1] + tri[3] + tri[5]) / 3
        for i in range(3):
            tri[2*i]   = _clamp(cx + factor * (tri[2*i]   - cx), 0, IMG_W-1)
            tri[2*i+1] = _clamp(cy + factor * (tri[2*i+1] - cy), 0, IMG_H-1)
        mutated[idx] = tri.astype(mutated.dtype)
    return mutated

def translate_triangle_mutation(indiv, mut_prob, delta=30):
    """Translates a random triangle (maintains shape, changes position)."""
    mutated = np.copy(indiv)
    if random.random() <= mut_prob:
        idx = random.randint(0, NUM_TRIANGLES - 1)
        tri = mutated[idx].copy()
        dx = random.randint(-delta, delta)
        dy = random.randint(-delta, delta)
        for i in range(3):
            tri[2*i]   = _clamp(tri[2*i]   + dx, 0, IMG_W-1)
            tri[2*i+1] = _clamp(tri[2*i+1] + dy, 0, IMG_H-1)
        mutated[idx] = tri
    return mutated


def uniform_triangle_crossover(parent1, parent2, crossover_prob, verbose=False):
    
    """
    Applies uniform crossover by swapping entire triangles between two parents.
    
    Instead of swapping bits, this method treats each triangle (6 coords + 3 colors)
    as a single unit (gene) and swaps them based on a 50% probability mask.

    Parameters:
        parent1 (np.ndarray): First parent matrix of shape (NUM_TRIANGLES, 10).
        parent2 (np.ndarray): Second parent matrix of shape (NUM_TRIANGLES, 10).
        crossover_prob (float): Probability of performing the crossover.
        verbose (bool): If True, prints the number of swapped triangles.

    Returns:
        tuple: Two new individuals (np.ndarray) as offspring.
    """

    if random.random() <= crossover_prob:
        # Create offspring by copying parents
        off1 = np.copy(parent1)
        off2 = np.copy(parent2)
        
        # Generate a random boolean mask to decide which triangles to swap
        # Each triangle has a 50% chance of being swapped
        mask = np.random.rand(NUM_TRIANGLES) < 0.5
        
        # Swap the triangles where the mask is True
        off1[mask] = parent2[mask]
        off2[mask] = parent1[mask]
        
        if verbose:
            print(f"Crossover performed: {mask.sum()} triangles swapped.")
            
        return off1, off2
    
    # If no crossover occurs, return exact copies of the original parents
    return np.copy(parent1), np.copy(parent2)


def one_point_crossover(parent1, parent2, crossover_prob):
    """
    Standard one-point crossover that splits the parents at a random point and swaps the tail segments.

    Parameters:
        parent1 (np.ndarray): First parent matrix of shape (NUM_TRIANGLES, 10).
        parent2 (np.ndarray): Second parent matrix of shape (NUM_TRIANGLES, 10).
        crossover_prob (float): Probability of performing the crossover.

    Returns:
        tuple: Two new individuals (np.ndarray) as offspring.
    """

    if random.random() <= crossover_prob:
        point = random.randint(1, NUM_TRIANGLES - 1)
        off1 = np.concatenate([parent1[:point], parent2[point:]])
        off2 = np.concatenate([parent2[:point], parent1[point:]])
        return off1, off2
    
    return np.copy(parent1), np.copy(parent2)

def two_point_crossover(parent1, parent2, crossover_prob):
    """
    Standard two-point crossover that splits the parents at two random points
    and swaps the middle segment between them.

    Parameters:
        parent1 (np.ndarray): First parent matrix of shape (NUM_TRIANGLES, 10).
        parent2 (np.ndarray): Second parent matrix of shape (NUM_TRIANGLES, 10).
        crossover_prob (float): Probability of performing the crossover.

    Returns:
        tuple: Two new individuals (np.ndarray) as offspring.
    """

    if random.random() <= crossover_prob:
        # Pick two distinct cut points
        point1 = random.randint(1, NUM_TRIANGLES - 2)
        point2 = random.randint(point1 + 1, NUM_TRIANGLES - 1)

        # Swap the middle segment between the two points
        off1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
        off2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])

        return off1, off2

    return np.copy(parent1), np.copy(parent2)

"""----------------------------------------------------------------------------------------------------------------------------------------------
# tirei da calculate_niche_counts

# Fitness Sharing Functions--------------------------------------------------------------------
def phenotypic_distance(img1, img2):


    Calculate Euclidean distance between two rendered images.
    Lower distance = more similar phenotypes.
    
    Args:
        img1, img2: numpy arrays of rendered images (same shape)
    
    Returns:
        float: Euclidean distance


    # Flatten images and compute L2 norm
    diff = img1.flatten() - img2.flatten()
    return np.sqrt(np.sum(diff ** 2))



def triangular_sharing_function(distance, niche_radius):


    Triangular sharing function: penalizes fitness based on distance to neighbors.
    
    sh(d) = 1 - (d / niche_radius)  if d < niche_radius
    sh(d) = 0                        if d >= niche_radius
    
    Args:
        distance (float): Distance to a neighbor
        niche_radius (float): Threshold distance for the niche
    
    Returns:
        float: Sharing value (0 to 1)


    if distance < niche_radius:
        return 1 - (distance / niche_radius)
    return 0

------------------------------------------------------------------------------------------------------------

"""


def calculate_niche_counts(population, niche_radius, rendered=None):

    """
    Calculate niche counts using GPU vectorized operations.

    Parameters
    ----------
    population : list or ndarray
        Population of individuals.

    niche_radius : float
        Radius used in the sharing function.

    rendered : cupy.ndarray or None
        Optional pre-rendered population tensor with shape (N, H, W, 4).

    Returns
    -------
    list[float]
        Niche count for each individual.
    """

    # Render only if necessary
    if rendered is None:
        rendered = render_population_cuda(population)

    # Flatten images
    n = rendered.shape[0]

    flat = rendered.reshape(n, -1).astype(cp.float32)

    # Compute pairwise Euclidean distances
    sq_norms = cp.sum(flat ** 2, axis=1)

    distances = cp.sqrt(
        cp.maximum(
            sq_norms[:, None] + sq_norms[None, :] - 2 * flat @ flat.T, 0))

    # Sharing function
    sharing_matrix = cp.maximum(0, 1 - distances / niche_radius)

    # Ignore self-distance
    cp.fill_diagonal(sharing_matrix, 0)

    niche_counts = sharing_matrix.sum(axis=1) + 1

    return cp.asnumpy(niche_counts).tolist()


def apply_fitness_sharing(raw_fitnesses, niche_counts):
    # tirei +1e-6 pq já fiz na calculate niche counts

    """
    Apply fitness sharing: shared_fitness = raw_fitness / niche_count.
    This penalizes individuals in crowded regions.
    
    Args:
        raw_fitnesses (list[float]): Raw fitness values
        niche_counts (list[float]): Niche count for each individual
    
    Returns:
        list[float]: Shared fitness values
    """
    
    return [fit * count for fit, count in zip(raw_fitnesses, niche_counts)]








