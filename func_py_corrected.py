#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        IMPORTS                                                                           #
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import random, copy
from sklearn.cluster import KMeans

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


def render(individual):
    # Create an empty transparent image (canvas) with RGBA channels
    img = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))

    # Decode each gene from [x1,y1,x2,y2,x3,y3,r,g,b,a] into triangle points [(x1,y1),(x2,y2),(x3,y3)] and RGBA color (r,g,b,a)
    for points, color in decode(individual):
        # Compute the polygon bounding box (smallest rectangle that contains the 3 points) so only the necessary area is processed.
        xs = [p[0] for p in points] # all the x values
        ys = [p[1] for p in points] # all the y values

        # Clamp the bounding box to the image boundaries.
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(IMG_W, int(max(xs)) + 1)
        y2 = min(IMG_H, int(max(ys)) + 1)

        # Skip polygons that are completely outside the image.
        if x1 >= x2 or y1 >= y2:
            continue

        # Convert points to local coordinates inside the smaller overlay.
        local_points = [(x - x1, y - y1) for x, y in points]

        # Create an overlay only as large as the bounding box, not the full image.
        overlay = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Draw the polygon onto the local overlay.
        draw.polygon(local_points, fill=color)

        # Alpha composite only the region where the polygon exists.
        img.alpha_composite(overlay, dest=(x1, y1))

    return np.array(img, dtype=np.float32)


def render_population_torch(population, device=None):
    
    """
    Render a full population of triangle-based individuals using PyTorch.

    Parameters:
        population:
            List or array with shape (POP_SIZE, NUM_TRIANGLES, 10).

            Each triangle/gene has:
            [x0, y0, x1, y1, x2, y2, R, G, B, A]

        device:
            PyTorch device. If None, uses CUDA when available.

    Returns:
        rendered:
            Torch tensor with shape (POP_SIZE, IMG_H, IMG_W, 4).
            Values are in [0, 255].
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert population to torch tensor on the specified device
    pop_tensor = torch.as_tensor(np.array(population), dtype=torch.float32, device=device)  # (POP_SIZE, NUM_TRIANGLES, 10)

    # Initialize an empty canvas for each individual
    canvas_rgb = torch.zeros((pop_tensor.shape[0], IMG_H, IMG_W, 3), dtype=torch.float32, device=device) # (POP_SIZE, H, W, 3) for RGB channels
    canvas_a = torch.zeros((pop_tensor.shape[0], IMG_H, IMG_W, 1), dtype=torch.float32, device=device) # (POP_SIZE, H, W, 1) for alpha channel

    # Create a coordinate grid for the image dimensions (H, W) 
    yy, xx = torch.meshgrid(torch.arange(IMG_H, device=device), torch.arange(IMG_W, device=device),
        indexing="ij") # ensure (H, W) order for correct broadcasting
    
    # Reshape coordinate grids to (1, H, W) for broadcasting with population of triangles
    xx = xx.float().unsqueeze(0)
    yy = yy.float().unsqueeze(0)

    # At each iteration, we process the t-th triangle of every individual. This means all individuals are rendered in parallel for that triangle index.
    for t in range(NUM_TRIANGLES):

        # Select triangle t from all individuals.
        tri = pop_tensor[:, t, :]

        # Extract triangle vertex coordinates.
        x0 = tri[:, 0].view(-1, 1, 1) # view(-1, 1, 1) reshapes to (POP_SIZE, 1, 1) for broadcasting, -1 means infer the batch size dimension
        y0 = tri[:, 1].view(-1, 1, 1)

        x1 = tri[:, 2].view(-1, 1, 1)
        y1 = tri[:, 3].view(-1, 1, 1)

        x2 = tri[:, 4].view(-1, 1, 1)
        y2 = tri[:, 5].view(-1, 1, 1)

        # Extract RGB and alpha values for the triangle.
        color = tri[:, 6:9].view(pop_tensor.shape[0], 1, 1, 3)
        alpha = (tri[:, 9].view(pop_tensor.shape[0], 1, 1, 1) / 255.0).clamp(0.0, 1.0)

        # Create a mask to check if a pixel is inside the triangle using the "Golden Triangle Test".
        # 1st: compute the barycentric coordinates of the pixel with respect to the triangle.
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom) # avoid division by zero for degenerate triangles

        cord1 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom 
        cord2 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom 
        cord3 = 1.0 - cord1 - cord2

        # 2nd: check if the barycentric coordinates are all between 0 and 1
        mask = (cord1 >= 0) & (cord2 >= 0) & (cord3 >= 0) & (cord1 <= 1) & (cord2 <= 1) & (cord3 <= 1) # (POP_SIZE, H, W) boolean mask where True means pixel is inside the triangle
        mask = mask.unsqueeze(-1) # (POP_SIZE, H, W, 1) to match color and alpha shapes

        # Compute the source alpha for the current triangle, which is the triangle's alpha value multiplied by the mask (1 inside the triangle, 0 outside).
        src_a = alpha * mask.float() 

        # New color = triangle color * transparency + old canvas color * part that is still visible
        # This simulates transparent triangles being layered on top of each other.
        canvas_rgb = color * src_a + canvas_rgb * (1.0 - src_a)

        # Update the accumulated alpha channel using the same logic.
        canvas_a = 255.0 * src_a + canvas_a * (1.0 - src_a)

    # Combine RGB and alpha into one RGBA tensor.
    rendered = torch.cat([canvas_rgb, canvas_a], dim=-1)

    return rendered


#Fitness Function
def population_fitness_rmse( population, target):

    """
    Renders each individual with the existing PIL render(),
    then calculates RMSE for the whole population using PyTorch.
    This approach leverages the GPU for efficient batch processing of the fitness evaluation, while still using the existing CPU-based rendering function.
    """

    # Determine the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move rendered images and target to GPU.
    rendered_tensor = render_population_torch(population, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)

    # Add population dimension to target to allow broadcasting during RMSE calculation:
    # target_tensor: ( H, W, C) --> (1, H, W, C), C=4 for RGBA.
    # rendered_tensor: (POP_SIZE, H, W, C)
    target_tensor = target_tensor.unsqueeze(0)

    # Compare rendered images to target --> (POP_SIZE, H, W, C) - (1, H, W, C) --> (POP_SIZE, H, W, C)
    diff = rendered_tensor - target_tensor

    # RMSE per individual
    rmse = torch.sqrt(torch.mean(diff * diff, dim=(1, 2, 3))) # dim=(1,2,3) means we average over H, W, C to get one RMSE value per individual in the population

    return rmse.detach().cpu().numpy().tolist()


# Selection Function------------------------------------------------------------------------------
def tournament_selection(population, fitnesses, k):

    """Select an individual from the population using tournament selection."""

    selected = random.sample(list(zip(population, fitnesses)), k) # randomly select k individuals
    selected.sort(key=lambda x: x[1]) # sort by fitness (lower is better)

    return copy.deepcopy(selected[0][0]) # return best individual's chromosome


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
    ]
    
    if weights is None:
        weights = [0.25, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10]  # default

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


# Fitness Sharing Functions--------------------------------------------------------------------
def phenotypic_distance(img1, img2):

    """
    Calculate Euclidean distance between two rendered images.
    Lower distance = more similar phenotypes.
    
    Args:
        img1, img2: numpy arrays of rendered images (same shape)
    
    Returns:
        float: Euclidean distance
    """

    # Flatten images and compute L2 norm
    diff = img1.flatten() - img2.flatten()
    return np.sqrt(np.sum(diff ** 2))


def triangular_sharing_function(distance, niche_radius):

    """
    Triangular sharing function: penalizes fitness based on distance to neighbors.
    
    sh(d) = 1 - (d / niche_radius)  if d < niche_radius
    sh(d) = 0                        if d >= niche_radius
    
    Args:
        distance (float): Distance to a neighbor
        niche_radius (float): Threshold distance for the niche
    
    Returns:
        float: Sharing value (0 to 1)
    """

    if distance < niche_radius:
        return 1 - (distance / niche_radius)
    return 0




def calculate_niche_counts(population, niche_radius):

    """
    Calculate the niche count for each individual.
    Niche count = sum of sharing function values with all other individuals.
    
    Args:
        population: list of individuals
        niche_radius (float): Niche radius parameter
    
    Returns:
        list[float]: Niche count for each individual
    """

    n = len(population)
    niche_counts = [0.0] * n
    rendered = [render(ind) for ind in population]
    for i in range(n):
        for j in range(i + 1, n):
            distance = phenotypic_distance(rendered[i], rendered[j])
            sharing  = triangular_sharing_function(distance, niche_radius)
            niche_counts[i] += sharing
            niche_counts[j] += sharing
    return [count + 1 for count in niche_counts]


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
    
    return [fit / (count + 1e-6) for fit, count in zip(raw_fitnesses, niche_counts)]




