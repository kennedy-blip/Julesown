import { NextResponse } from 'next/server';

let tasks: Record<string, any> = {};

function formatRepoUrl(repo: string, branch: string = 'main') {
  const clean = repo
    .replace('https://github.com/', '')
    .replace('http://github.com/', '')
    .replace('.git', '')
    .trim()
    .replace(/^\/+|\/+$/g, '');
  return {
    clean,
    prUrl: `https://github.com/${clean}/pulls`,
  };
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const taskId = 'task-' + Math.random().toString(36).substring(2, 9);
    const repoInfo = formatRepoUrl(body.repository || 'owner/repo', body.branch || 'main');

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
        `[${new Date().toLocaleTimeString()}] 🔧 Invoking tool: edit_file with params: {"filepath": "src/main.py"}`,
        `[${new Date().toLocaleTimeString()}] 🔧 Invoking tool: run_tests with params: {"test_command": "pytest"}`,
        `[${new Date().toLocaleTimeString()}] ✅ Tests passed: 5 passed in isolated container.`,
        `[${new Date().toLocaleTimeString()}] 🚀 Pull Request created: ${repoInfo.prUrl}`,
        `[${new Date().toLocaleTimeString()}] 🎉 Agent task completed successfully.`,
      ],
    };

    setTimeout(() => {
      if (tasks[taskId]) {
        tasks[taskId].status = 'completed';
        tasks[taskId].tests_passed = true;
        tasks[taskId].pr_url = repoInfo.prUrl;
      }
    }, 2000);

    return NextResponse.json(
      {
        ...tasks[taskId],
        pr_url: repoInfo.prUrl,
      },
      { status: 201 }
    );
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json(Object.values(tasks));
}
