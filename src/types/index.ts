export interface Project {
  id: number
  title: string
  category: string
  description: string
  imageUrl: string
  tech: string[]
  liveUrl?: string
  githubUrl?: string
}
export interface Experience {
  position: string
  company: string
  description: string
  start_date: string
  end_date?: string
  current?: boolean
}

export interface Skill {
  name: string
  category: string
  proficiency: number
  icon: string
}

export interface SkillCategory {
  title: string
  skills: string[]
}
