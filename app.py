from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import base64
import time
import io
import sys

app = Flask(__name__)

sys.setrecursionlimit(2000)

# Bubble Sort
def bubble_sort(arr, visualize_steps=False):
    arr = arr.copy()
    n = len(arr)
    start_time = time.perf_counter()
    steps = []

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            if visualize_steps:
                steps.append(arr.copy())

    full_time = time.perf_counter() - start_time

    if visualize_steps:
        images = create_plot(steps)
        return full_time * 1000, images
    return full_time * 1000

def merge_sort(arr, visualize_steps=False):
    arr = arr.copy()
    steps = []

    def merge_sort_helper(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            merge_sort_helper(arr, left, mid)
            merge_sort_helper(arr, mid + 1, right)
            merge(arr, left, mid, right)

    def merge(arr, left, mid, right):
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        i = j = 0
        k = left

        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
            if visualize_steps:
                steps.append(arr.copy())

        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
            if visualize_steps:
                steps.append(arr.copy())

        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
            if visualize_steps:
                steps.append(arr.copy())

    start_time = time.perf_counter()
    merge_sort_helper(arr, 0, len(arr) - 1)
    total_time = time.perf_counter() - start_time
    
    if visualize_steps:
        images = create_plot(steps)
        return total_time * 1000, images
    return total_time * 1000

def quick_sort(arr, visualize_steps=False):
    arr = arr.copy()
    steps = []

    def quick_sort_helper(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            if visualize_steps:
                steps.append(arr.copy())
            quick_sort_helper(arr, low, pi - 1)
            quick_sort_helper(arr, pi + 1, high)

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                if visualize_steps:
                    steps.append(arr.copy())
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        if visualize_steps:
            steps.append(arr.copy())
        return i + 1

    start_time = time.perf_counter()
    quick_sort_helper(arr, 0, len(arr) - 1)
    total_time = time.perf_counter() - start_time
    
    if visualize_steps:
        images = create_plot(steps)
        return total_time * 1000, images
    return total_time * 100

def radix_sort(arr, visualize_steps = False):
    arr = arr.copy()
    max_value = max(arr)
    exp = 1
    start_time = time.perf_counter()
    steps = []

    while max_value // exp > 0:
        counting_sort(arr, exp)
        if visualize_steps:
            steps.append(arr.copy())
        exp *= 10

    total_time = time.perf_counter() - start_time
    
    if visualize_steps:
        images = create_plot(steps)
        return total_time * 1000, images
    return total_time * 1000

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def linear_search(arr, target, visualize_steps=False):
    steps = []  # Step tracking for visualization

    for i, x in enumerate(arr):
        if visualize_steps:
            # Record the current array and index being checked
            step = {
                'array': arr.copy(),
                'current_index': i,
                'target': target
            }
            steps.append(step)  # Append current step

        if x == target:
            # Return both the index and the steps if visualizing is true
            return i, steps if visualize_steps else i  

    # Return -1 if not found, with steps for visualization
    return -1, steps if visualize_steps else -1

# Function to create linear search visualization (New function for linear search)
def create_linear_search_plot(steps):
    images = []
    
    for i, step in enumerate(steps):
        arr = step['array']
        current_index = step['current_index']
        target = step['target']

        fig, ax = plt.subplots()
        bars = ax.bar(range(len(arr)), arr, color='blue')
        bars[current_index].set_color('red')  # Highlight current index in red

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"Step {i}: Searching for {target} (Current Index: {current_index})")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        images.append(f"data:image/png;base64,{image_base64}")
    
    return images

def linear_search_performance():
    arr = np.random.randint(0, 100, size=50)

    target = np.random.choice(arr)

    start_time_unsorted = time.perf_counter()
    indices_unsorted = []
    for i in range(len(arr)):
        if arr[i] == target:
            indices_unsorted.append(i)
    
    time_unsorted = (time.perf_counter() - start_time_unsorted) * 1000

    sorted_arr = sorted(arr.tolist())

    indices_sorted = []
    start_time_sorted = time.perf_counter()
    for i in range(len(sorted_arr)):
        if sorted_arr[i] == target:
            indices_sorted.append(i)
        elif sorted_arr[i] > target:
            break 
    time_sorted = (time.perf_counter() - start_time_sorted) * 1000

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(['Unsorted Array', 'Sorted Array'], [time_unsorted, time_sorted], color=['blue', 'green'])
    ax.set_title('Linear Search Performance: Unsorted vs. Sorted Array')
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Array Type')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    image_url = f"data:image/png;base64,{image_base64}"

    return image_url, arr, sorted_arr, target

def create_plot(steps):
    images = []
    for i, step in enumerate(steps):
        fig, ax = plt.subplots()
        ax.bar(range(len(step)), step, color='blue')
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"Step {i}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        images.append(f"data:image/png;base64,{image_base64}")

    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    if request.method == 'POST':
        size = int(request.form['size'])
        action = request.form['action']

        if action == 'performance':
            plot_img, performance_data = create_performance_plot(size)
            df = pd.DataFrame(list(performance_data.items()), columns=['Algorithm', 'Time (ms)'])
            table_html = df.to_html(classes='data-table', index=False)

            return render_template('visualization.html', plot_img=plot_img, table_html=table_html)

@app.route('/visualize_animation', methods=['POST'])
def visualize_animation():
    if request.method == 'POST':
        size = int(request.form['size'])
        selected_algorithms = request.form.getlist('algorithms')

        all_images = {}
        arr = random.sample(range(10, 100), size)

        for algorithm in selected_algorithms:
            if algorithm == 'Bubble Sort':
                images = bubble_sort(arr, visualize_steps=True)[1]
            elif algorithm == 'Merge Sort':
                images = merge_sort(arr, visualize_steps=True)[1]
            elif algorithm == 'Quick Sort':
                images = quick_sort(arr, visualize_steps=True)[1]
            elif algorithm == 'Radix Sort':
                images = radix_sort(arr, visualize_steps=True)[1]

            all_images[algorithm] = images

        return render_template('visualize_animation.html', all_images=all_images)

@app.route('/visualize_linear_search', methods=['POST'])
def visualize_linear_search():
    if request.method == 'POST':
        size = 25
        arr = random.sample(range(10, 1000), size)
        target = arr[20]
        _, steps = linear_search(arr, target, visualize_steps=True)
        images = create_linear_search_plot(steps)

        return render_template('visualize_animation.html', all_images={'Linear Search': images})

@app.route('/visualize_linear_search_performance', methods=['POST'])
def visualize_linear_search_performance():
    if request.method == 'POST':
        image_url, arr, sorted_arr, target = linear_search_performance()
        return render_template(
            'visualize_linear_search_performance.html',
            performance=image_url,
            unsorted_array=arr.tolist(),
            sorted_array=sorted_arr,
            target_value=target
        )

if __name__ == '__main__':
    app.run(debug=True)