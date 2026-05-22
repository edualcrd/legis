-- Esquema de autenticación beta de Legis (magic link).
-- Ejecutar una sola vez en el editor SQL de Supabase (proyecto del beta).
--
-- El backend (src/auth.py) accede con la SERVICE_ROLE key, que ignora RLS.
-- Esa key es solo server-side y nunca llega al frontend. Por eso no definimos
-- políticas RLS aquí: ningún cliente público toca estas tablas directamente.

-- Usuarios autorizados a entrar al beta.
create table if not exists usuarios (
    id         uuid primary key default gen_random_uuid(),
    email      text unique not null,
    plan       text not null default 'beta',
    activo     boolean not null default true,
    created_at timestamptz not null default now()
);

-- Tokens de un solo uso del magic link. Cada solicitud de acceso genera uno
-- con expiración de 24 horas; se invalida al usarse (usado = true).
create table if not exists magic_tokens (
    id         uuid primary key default gen_random_uuid(),
    email      text not null,
    token      text unique not null,
    usado      boolean not null default false,
    expira_at  timestamptz not null,
    created_at timestamptz not null default now()
);

-- Búsqueda por token en cada verificación del magic link.
create index if not exists idx_magic_tokens_token on magic_tokens (token);

-- (Opcional) Limpieza manual de tokens vencidos:
--   delete from magic_tokens where expira_at < now();
