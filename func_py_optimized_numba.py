#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        IMPORTS                                                                           #
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import random
import cupy as cp
from numba import cuda

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        PARAMETERS                                                                        #
#------------------------------------------------------------------------------------------------------------------------------------------#

IMG_W, IMG_H    = 300, 400
NUM_TRIANGLES   = 100


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


#----------------------------------------------------------------------------------------------------------

# RENDERING FUNCTIONS


@cuda.jit
def render_triangles_cuda(output, triangles, bboxes, pop_size):

    """
        CUDA kernel that renders a population of triangle-based images in parallel.

        Each CUDA thread is responsible for computing the RGBA value of a single
        pixel (x, y) for one individual in the population. The kernel iterates
        through all triangles belonging to that individual and performs alpha
        blending to accumulate the final pixel color.

        Rendering process:
            1. Determine the thread's pixel center coordinates and individual index.
            2. Reject threads outside image/population bounds.
            3. For each triangle:
                - Check whether the pixel lies inside the triangle bounding box.
                - Compute triangle area.
                - Compute barycentric coordinates.
                - Determine whether the pixel is inside the triangle.
                - Blend the triangle color into the accumulated pixel color
                using alpha compositing.
            4. Store the final RGBA value in the output.

        ------------------------------
        Parameters:

            output [cuda.device_array]: image buffer with shape (pop_size, IMG_H, IMG_W, 4); stores 
        the rendered RGBA values for every individual.

            triangles [cuda.device_array]: triangle data for the population with shape (pop_size, NUM_TRIANGLES, 10)

            bboxes [cuda.device_array]: precomputed axis-aligned bounding boxes for each triangle with shape
        (pop_size, NUM_TRIANGLES, 4); bounding box format: [minX, minY, maxX, maxY] 


            pop_size [int]: number of individuals/images being rendered.

    """

    # thread indices: x -> pixel x-coord.; y -> pixel y-coord.; indiv_idx -> inddex of individual in the population
    x, y, indiv_idx = cuda.grid(3)

    # ignore threads outside valid image or population bounds
    if x >= IMG_W or y >= IMG_H or indiv_idx >= pop_size:
        return

    # centering pixel coords.
    cx = x + 0.5
    cy = y + 0.5

    # accumulated RGBA color for this pixel (initialized as 0)
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0
    acc_a = 0.0

    # iterate through every triangle of the current individual
    for t in range(NUM_TRIANGLES):

        # bounding box of current triangle
        minX = bboxes[indiv_idx, t, 0]
        minY = bboxes[indiv_idx, t, 1]
        maxX = bboxes[indiv_idx, t, 2]
        maxY = bboxes[indiv_idx, t, 3]

        # reject pixels outside bbox
        if cx < minX or cx > maxX or cy < minY or cy > maxY:
            continue

        # triangle data (x1, y1, x2, y2, x3, y3, R, G, B, A)
        tri = triangles[indiv_idx, t]

        # triangle vertices
        x0 = tri[0]
        y0 = tri[1]

        x1 = tri[2]
        y1 = tri[3]

        x2 = tri[4]
        y2 = tri[5]

        # normalizing alpha
        alpha = tri[9] / 255.0

        if alpha < 0.0:
            alpha = 0.0

        if alpha > 1.0:
            alpha = 1.0

        # skipping fully transparent triangles
        if alpha <= 0.0:
            continue

        # triangle area
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

        abs_area = abs(area)

        # skip excessively small triangles
        if abs_area < 0.2:
            continue

        # barycentric coordinates used to determine whether the pixel lies inside the triangle
        bc0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / abs_area
        bc1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / abs_area
        bc2 = 1.0 - bc0 - bc1

        # if one of the barycentric coords. is negative the pixel is outside the triangle and that triangle is skipped
        if bc0 < 0.0 or bc1 < 0.0 or bc2 < 0.0:
            continue

        # alpha blending weights
        inv_alpha = 1.0 - alpha

        # cumulative pixel RGBA values (previous values have inv_alpha weight and new ones have alpha weight)
        acc_r = tri[6] * alpha + acc_r * inv_alpha
        acc_g = tri[7] * alpha + acc_g * inv_alpha
        acc_b = tri[8] * alpha + acc_b * inv_alpha
        acc_a = 255.0 * alpha + acc_a * inv_alpha

    # store final RGBA pixel value
    output[indiv_idx, y, x, 0] = acc_r
    output[indiv_idx, y, x, 1] = acc_g
    output[indiv_idx, y, x, 2] = acc_b
    output[indiv_idx, y, x, 3] = acc_a





def compute_bboxes(population):
    '''
        For every triangle in the population, this function computes the smallest axis-aligned rectangle
    that contains said triangle, known as its bounding box.

    --------------
        Parameters:
            population [array]: (pop_size, NUM_TRIANGLES, TRIANGLE_DATA)
        
        Returns:
            bboxes [array]: bounding boxes with shape (pop_size, NUM_TRIANGLES, 4) 

    '''

    population = np.asarray(population, dtype=np.float32)

    bboxes = np.zeros(
        (population.shape[0], NUM_TRIANGLES, 4),
        dtype=np.float32
    )

    x = population[:, :, [0, 2, 4]]
    y = population[:, :, [1, 3, 5]]

    bboxes[:, :, 0] = x.min(axis=2)  # minX
    bboxes[:, :, 1] = y.min(axis=2)  # minY
    bboxes[:, :, 2] = x.max(axis=2)  # maxX
    bboxes[:, :, 3] = y.max(axis=2)  # maxY

    return bboxes





def render_population_cuda(population):
    '''
        Renders an entire population of triangle-based images on the GPU.

        This function prepares the population data, computes triangle bounding
    boxes, transfers all required data to the GPU, launches the CUDA rendering
    kernel, and returns the rendered images as a CuPy array.

        Rendering workflow:
        1. Get the population size (N).
        2. Compute bounding boxes for all triangles.
        3. Transfer triangle and bounding box data to GPU memory.
        4. Allocate an output image buffer on the GPU.
        5. Configure CUDA execution parameters.
        6. Launch the rendering kernel.
        7. Return the rendered images.

    ---------------------
        Parameters:
            population [array]: population with shape (N, NUM_TRIANGLES, TRIANGLE_DATA)
        where TRIANGLE_DATA has shape [x0, y0, x1, y1, x2, y2, r, g, b, alpha]

        Returns: 
            cupy.ndarray: Rendered RGBA images stored on the GPU with shape (N, IMG_H, IMG_W, 4)

    '''

    population = np.asarray(population, dtype=np.float32)

    N = population.shape[0]

    # bounding boxes
    bboxes = compute_bboxes(population)

    # move to GPU using numba
    triangles_gpu = cuda.to_device(population)
    bboxes_gpu = cuda.to_device(bboxes)

    # output buffer
    output_gpu = cuda.device_array(
        (N, IMG_H, IMG_W, 4),
        dtype=np.float32)

    # threads per block
    threads = (16, 16, 1)

    # blocks per grid
    blocks = (
        (IMG_W + threads[0] - 1) // threads[0],
        (IMG_H + threads[1] - 1) // threads[1],
        N)

    # launch kernel
    render_triangles_cuda[blocks, threads](
        output_gpu,
        triangles_gpu,
        bboxes_gpu,
        N)

    # return GPU array
    return cp.asarray(output_gpu)


def render(individual):
    '''
        Renders a single triangle-based individual into an RGBA image.

        Since the GPU rendering pipeline expects a population of individuals, the input individual is 
    temporarily expanded into a population of size 1.

    ---------------------
        Params:
            individual [array]: Triangle representation of a single individual with shape (NUM_TRIANGLES, TRIANGLE_DATA)

        Returns:
            np.ndarray: rendered RGBA image with shape (IMG_H, IMG_W, 4)
    '''

    # adding a batch dimension so the individual becomes a population of size 1
    population = np.expand_dims(individual, axis=0)

    # render using CUDA rendering pipeline
    rendered = render_population_cuda(population)

    # remove batch dimension and transfer image from GPU to CPU memory
    img = cp.asnumpy(rendered[0])

    return img.astype(np.float32)



# -----------------------------------------------------------------------------------------------


# Fitness Function--------------------------------------------------------------------------------

def population_fitness_rmse(rendered_population, target):
    '''
        Computes the RMSE (Root Mean Squared Error) fitness for a population of rendered images compared
    to a target image. 

    ------------------
        Parameters:
            rendered_population [cupy.ndarray]: Batch of rendered images with shape (N, IMG_H, IMG_W, 4)
            target [array]: target image to compare against
        
            Returns:
            list[float]: list containing RMSE for each individual of the population
    '''

    # transfering the target to gpu
    target_gpu = cp.asarray(target, dtype=cp.float32)

    # pixewise difference between image and target
    diff = rendered_population - target_gpu[None, :, :, :]
    # RMSE computation
    rmse = cp.sqrt(cp.mean(diff * diff, axis=(1,2,3)))

    # list of RMSE values for the population
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

