declare module '../data/projects' {
  export interface Project {
    slug: string;
    title: string;
    description: string;
    tech_stack: string[];
    created_date: string;
    featured?: boolean;
    github_url?: string;
    demo_url?: string | null;
    image_name?: string;
    detailed_description?: string;
    challenges?: string;
    results?: string;
  }
  export const projects: Project[];
}
