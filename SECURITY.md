# Security policy

Do not open a public issue containing a vulnerability, credential, private video,
dataset token or proprietary checkpoint. Use GitHub's private vulnerability
reporting feature for this repository when enabled by its owner. If it is not
enabled, contact the repository owner privately through their published GitHub
profile contact channel.

Only the latest revision on the default branch receives security fixes. This is a
research alpha and should not be exposed directly to the public internet. The
default Compose port is intended for localhost development; configure an
authenticated reverse proxy, restricted CORS origins and network policy for any
shared deployment.

Before reporting, remove raw footage, crops, prompts containing personal data,
tokens and filesystem paths from logs. Never commit `.env`, `weights/`, `data/`
runtime contents or `runs/` artifacts.
