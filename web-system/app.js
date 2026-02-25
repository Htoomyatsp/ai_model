const storageKey = "web_system_tasks_v1";
const form = document.getElementById("task-form");
const titleInput = document.getElementById("task-title");
const ownerInput = document.getElementById("task-owner");
const taskList = document.getElementById("task-list");
const clearAllBtn = document.getElementById("clear-all");

function loadTasks() {
  try {
    return JSON.parse(localStorage.getItem(storageKey)) || [];
  } catch {
    return [];
  }
}

function saveTasks(tasks) {
  localStorage.setItem(storageKey, JSON.stringify(tasks));
}

function toggleDone(id) {
  const tasks = loadTasks().map((task) =>
    task.id === id ? { ...task, done: !task.done } : task
  );
  saveTasks(tasks);
  render();
}

function removeTask(id) {
  const tasks = loadTasks().filter((task) => task.id !== id);
  saveTasks(tasks);
  render();
}

function clearAll() {
  saveTasks([]);
  render();
}

function render() {
  const tasks = loadTasks();
  taskList.innerHTML = "";

  if (!tasks.length) {
    const empty = document.createElement("li");
    empty.textContent = "No tasks yet.";
    empty.className = "task-meta";
    taskList.appendChild(empty);
    return;
  }

  for (const task of tasks) {
    const item = document.createElement("li");
    item.className = "task-item";

    const left = document.createElement("div");
    const title = document.createElement("div");
    title.textContent = task.done ? `✓ ${task.title}` : task.title;
    title.style.fontWeight = "600";
    title.style.textDecoration = task.done ? "line-through" : "none";

    const meta = document.createElement("div");
    meta.className = "task-meta";
    meta.textContent = `Owner: ${task.owner} | Created: ${new Date(task.createdAt).toLocaleString()}`;

    left.appendChild(title);
    left.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "task-actions";

    const doneBtn = document.createElement("button");
    doneBtn.textContent = task.done ? "Undo" : "Done";
    doneBtn.addEventListener("click", () => toggleDone(task.id));

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "delete";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => removeTask(task.id));

    actions.appendChild(doneBtn);
    actions.appendChild(deleteBtn);

    item.appendChild(left);
    item.appendChild(actions);
    taskList.appendChild(item);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const title = titleInput.value.trim();
  const owner = ownerInput.value.trim();

  if (!title || !owner) {
    return;
  }

  const tasks = loadTasks();
  tasks.push({
    id: crypto.randomUUID(),
    title,
    owner,
    done: false,
    createdAt: Date.now(),
  });

  saveTasks(tasks);
  form.reset();
  render();
});

clearAllBtn.addEventListener("click", clearAll);

render();
