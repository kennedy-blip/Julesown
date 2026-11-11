import { NextResponse } from 'next/server';

// Simulation store for local development or when backend is served as route handler
let tasks: Record<string, any> = {};

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const taskId = 'task-' + Math.random().toString(36).substring(2, 9);

    tasks[taskId] = {
      task_id: taskId,
      repository: body.repository,
      branch: body.branch || 'main',
      prompt: body.prompt,
      status: 'running',
      pr_url: null,
      tests_passed: null,
      logs: [
        `[${new Date().toLocaleTimeString()}] Initializing E2B Linux MicroVM sandbox...`,
        `[${new Date().toLocaleTimeString()}] Cloning repository: ${body.repository}`,
        `[${new Date().toLocaleTimeString()}] Analyzing prompt: "${body.prompt}"`,
        `[${new Date().toLocaleTimeString()}] 🔧 Invoking tool: view_file with params: {"filepath": "README.md"}`,
        `[${new Date().toLocaleTimeString()}] 🔧 Invoking tool: edit_file with params: {"filepath": "src/fix.py"}`,
        `[${new Date().toLocaleTimeString()}] 🔧 Invoking tool: run_tests with params: {"test_command": "pytest"}`,
        `[${new Date().toLocaleTimeString()}] ✅ Tests passed: 5 passed, 0 failures.`,
        `[${new Date().toLocaleTimeString()}] 🚀 Pull Request created: https://github.com/org/repo/pull/42`,
        `[${new Date().toLocaleTimeString()}] 🎉 Agent task completed successfully.`,
      ],
    };

    // Simulate completion
    setTimeout(() => {
      if (tasks[taskId]) {
        tasks[taskId].status = 'completed';
        tasks[taskId].tests_passed = true;
        tasks[taskId].pr_url = 'https://github.com/org/repo/pull/42';
      }
    }, 2500);

    return NextResponse.json(tasks[taskId], { status: 201 });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json(Object.values(tasks));
}
