# Agent Skills

`nas.adk.skills` supports portable `SKILL.md` packages without depending on Nexus or a particular database or sandbox. Skills augment an existing agent and Runner; they do not choose models or grant tools.

## A filesystem-backed agent

With an already configured model:

```python
from nas.adk import Agent, Runner
from nas.adk.skills import DirectorySkillStore, SkillRuntime

store = DirectorySkillStore("./skills")
runner = Runner(
    agent=Agent(name="assistant", model=model, instruction="Help the user."),
    skills=SkillRuntime(store),
)
async for event in runner.run_async("user", "session", "Prepare a weekly report"):
    print(event.type, event.content)
```

The catalog exposes only names and descriptions initially. `activate_skill` loads the procedure, and `read_skill_file` reads resources with pagination. The Runner protects active procedures during context truncation and emits `skill` activation events. Explicit activation uses `SkillRuntime(store, explicit=[skill_reference_id])`.

The runtime pins catalog references in session state. A restored session must receive the same authorized catalog, including version digests; it never silently substitutes newer packages. Create a runtime for the authorized execution scope. Child agents do not automatically inherit its skills or executor.

## Import and host adapters

Public exports include `SkillManifest`, `SkillPackage`, `SkillRef`, `package_from_files`, `import_directory`, `import_zip`, `import_github` and `export_zip`. File bytes are base64 encoded in the JSON contract, with a digest over the canonical file mapping. Validation verifies that metadata and instructions match `SKILL.md`.

ZIPs are bounded, cannot contain traversal paths, duplicate entries or symlinks, and may contain multiple skill directories. A package is limited to 256 files and 10 MB. GitHub import accepts repository or `/tree/ref/path` URLs and requires an injected async fetcher `(url, maximum_bytes) -> bytes`. Folder URLs fetch only selected immutable blobs; unrelated large files are not downloaded. The host enforces its network policy and streaming size limit. Import does not execute scripts or install dependencies.

Applications can implement:

```python
class SkillStore(Protocol):
    async def list(self) -> list[SkillRef]: ...
    async def load(self, id: str) -> SkillPackage: ...

class SkillExecutor(Protocol):
    async def execute(self, ref: SkillRef, path: str, args: list[str], ctx: Context): ...
```

Only an explicitly supplied executor enables `run_skill_script`. The ADK does not provide an implicit local subprocess executor for external packages. All skill tools use the Runner's normal validation, policy callbacks, budgets and interruption handling. `allowed-tools` metadata does not alter the configured tool set or policy.

## Creation, checkpoints and evaluation

`create_skill_creator(model, backend)` builds an authoring assistant using `SkillDraftBackend`: list/read draft files, write/delete files and validate. The host owns draft persistence, test execution, publishing and gallery sharing. Binary assets remain host-owned; the creator reads text resources on demand.

`SessionCheckpoint.from_session(session)` exports versioned events and state. Serialize it with `model_dump_json()` and restore it into the same application, user and session with `checkpoint.restore(session)`. Credentials and executor objects are not part of the checkpoint.

`EvalRunner(..., execution=...)` accepts an async event iterator for each `EvalCase`, so a host can execute cases through its own authenticated, durable harness. `EvalRunner.assess_observation` scores recorded outputs while preserving pending and failed states. Text containment checks are deterministic checks, rather than a semantic quality assessment.
